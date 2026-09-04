import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import label

class PCSWPromptGenerator(nn.Module):
    def __init__(self, feature_channels=256, num_classes=1, threshold_tau=0.8):
        super().__init__()
        self.threshold_tau = threshold_tau
        # 公式 (17), (18): Conv + Softmax/Sigmoid
        self.mask_head = nn.Sequential(
            nn.Conv2d(feature_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        self.num_classes = num_classes

    def forward_coarse_mask(self, multi_scale_feats):

        f_high = multi_scale_feats[0]  # [S, C, H, W]
        logits = self.mask_head(f_high)
        if self.num_classes == 1:
            masks = torch.sigmoid(logits)
        else:
            masks = torch.softmax(logits, dim=1)
        return masks

    def evaluate_continuity_and_filter(self, binary_volume_np):

        # 3D 8-连通性 (26-连通在3D结构中，对应 scipy 默认 structure 生成)
        labeled_arr, num_features = label(binary_volume_np)
        if num_features == 0:
            return None, 0.0

        counts = np.bincount(labeled_arr.flat)
        counts[0] = 0  # 忽略背景
        max_component_id = counts.argmax()
        max_vol_size = counts[max_component_id]
        total_pixels = binary_volume_np.sum()

        if total_pixels == 0:
            return None, 0.0

        C = max_vol_size / float(total_pixels)
        clean_mask = (labeled_arr == max_component_id)
        return clean_mask, C

    @torch.no_grad()
    def generate_prompts(self, coarse_masks, orig_img_shape):

        S, _, H, W = coarse_masks.shape
        coarse_bin = (coarse_masks > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)

        # 窗口大小区间定义为 [S/3, S/2]，从 median 处启发式搜索
        N_min = max(2, S // 3)
        N_max = max(3, S // 2)
        N_init = (N_min + N_max) // 2

        best_slice_idx = None
        best_prompt_box = None
        best_prompt_point = None


        for N in range(N_init, N_max + 1):
            found_valid = False
            for i in range(0, S - N + 1):
                sub_vol = coarse_bin[i : i + N]
                clean_vol, C = self.evaluate_continuity_and_filter(sub_vol)

                if clean_vol is not None and C >= self.threshold_tau:

                    slice_areas = clean_vol.sum(axis=(1, 2))
                    local_best = np.argmax(slice_areas)
                    best_slice_idx = i + local_best
                    valid_slice_mask = clean_vol[local_best]

                    y_indices, x_indices = np.where(valid_slice_mask > 0)
                    if len(y_indices) > 0:
                        y_min, y_max = y_indices.min(), y_indices.max()
                        x_min, x_max = x_indices.min(), x_indices.max()
                        best_prompt_box = np.array([x_min, y_min, x_max, y_max])
                        best_prompt_point = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0])
                    found_valid = True
                    break
            if found_valid:
                break


        if best_slice_idx is None:
            sum_areas = coarse_bin.sum(axis=(1, 2))
            best_slice_idx = int(np.argmax(sum_areas))
            y_indices, x_indices = np.where(coarse_bin[best_slice_idx] > 0)
            if len(y_indices) > 0:
                best_prompt_box = np.array([x_indices.min(), y_indices.min(), x_indices.max(), y_indices.max()])
                best_prompt_point = np.array([np.mean(x_indices), np.mean(y_indices)])
            else:
                best_prompt_box = np.array([0, 0, W, H])
                best_prompt_point = np.array([W / 2.0, H / 2.0])

        return {
            "prompt_slice_idx": best_slice_idx,
            "point": torch.tensor(best_prompt_point, dtype=torch.float32).unsqueeze(0), # [1, 2]
            "box": torch.tensor(best_prompt_box, dtype=torch.float32).unsqueeze(0)        # [1, 4]
        }