import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=512, out_dim=256):
        super().__init__()
        # 论文 4.2 节：4 linear layers [256, 512, 512, 256] with LayerNorm and ReLU
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        # x: [B, C, H, W] -> [B, H, W, C] -> MLP -> [B, C, H, W]
        B, C, H, W = x.shape
        x_perm = x.permute(0, 2, 3, 1)
        out = self.net(x_perm)
        return out.permute(0, 3, 1, 2)


class SingleScaleDFE(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.mlp = MLP(in_dim=channels, hidden_dim=512, out_dim=channels)

        self.conv_align = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, F_s1, F_s2):

        F_s1 = self.conv_align(F_s1)
        F_s2 = self.conv_align(F_s2)


        F_s1_vec = F_s1.mean(dim=(2, 3))
        F_s2_vec = F_s2.mean(dim=(2, 3))


        # S_cosine: [B, 1, 1, 1]
        sim = F.cosine_similarity(F_s1_vec, F_s2_vec, dim=1).view(-1, 1, 1, 1)


        w = torch.sigmoid(sim)
        F_fused = w * self.mlp(F_s1) + (1.0 - w) * self.mlp(F_s2)


        delta_F = (1.0 - sim) * (F_s2 - F_s1)
        F_adjusted = F_fused + delta_F

        return F_adjusted


class DFEMechanism(nn.Module):

    def __init__(self, in_channels_list=[64, 128, 256], out_dim=256):
        super().__init__()
        self.proj_convs = nn.ModuleList([
            nn.Conv2d(c, out_dim, kernel_size=1) if c != out_dim else nn.Identity()
            for c in in_channels_list
        ])
        self.dfe_modules = nn.ModuleList([
            SingleScaleDFE(channels=out_dim) for _ in in_channels_list
        ])

    def forward(self, feats_s1, feats_s2):
        """
        feats_s1, feats_s2: list of multi-scale tensors from image encoder
        """
        enhanced_feats = []
        for i, (f1, f2) in enumerate(zip(feats_s1, feats_s2)):
            p1 = self.proj_convs[i](f1)
            p2 = self.proj_convs[i](f2)
            f_enh = self.dfe_modules[i](p1, p2)
            enhanced_feats.append(f_enh)
        return enhanced_feats