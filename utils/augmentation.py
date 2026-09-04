import torch
import torchvision.transforms as T
import random

class WeakAugmentation:
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, img):
        if random.random() > 0.5:
            img = torch.flip(img, dims=[-1])
        return img


class StrongAugmentation:
    def __init__(self):
        self.color_jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2)
        self.blur = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))

    def __call__(self, img):
        # 强增强：Color Jittering, Grayscaling, Gaussian Blurring
        x = img.clone()
        if random.random() > 0.2:
            x = self.color_jitter(x)
        if random.random() > 0.3:
            # 灰度化变换 (3 通道保持一致)
            gray = x.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            x = gray
        if random.random() > 0.5:
            x = self.blur(x)
        return x


def apply_cutmix(img1, img2, alpha=1.0):

    lam = np.random.beta(alpha, alpha)
    B, C, H, W = img1.shape
    bbx1, bby1, bbx2, bby2 = rand_bbox(H, W, lam)
    img1[:, :, bbx1:bbx2, bby1:bby2] = img2[:, :, bbx1:bbx2, bby1:bby2]
    return img1

def rand_bbox(h, w, lam):
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    return bbx1, bby1, bbx2, bby2