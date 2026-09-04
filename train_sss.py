import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sam2_train.build_sam import build_sam2_video_predictor
from models.dfe import DFEMechanism
from models.pcsw import PCSWPromptGenerator
from utils.augmentation import WeakAugmentation, StrongAugmentation

def compute_supervised_loss(pred_mask, gt_mask):

    bce = F.binary_cross_entropy_with_logits(pred_mask, gt_mask)
    pred_prob = torch.sigmoid(pred_mask)
    intersection = (pred_prob * gt_mask).sum()
    dice = 1.0 - (2. * intersection + 1e-5) / (pred_prob.sum() + gt_mask.sum() + 1e-5)
    return bce + dice

def train_one_epoch(
    sam2_model,
    dfe_module,
    pcsw_generator,
    labeled_loader,
    unlabeled_loader,
    optimizer,
    device,
    epoch
):
    sam2_model.train()
    dfe_module.train()
    pcsw_generator.train()

    weak_aug = WeakAugmentation()
    strong_aug = StrongAugmentation()

    unlabeled_iter = iter(unlabeled_loader)

    for step, (imgs_l, masks_l) in enumerate(labeled_loader):

        imgs_l = imgs_l.to(device)    # [B, T, C, H, W] 或 [B, C, H, W]
        masks_l = masks_l.to(device)

        try:
            imgs_u = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            imgs_u = next(unlabeled_iter)

        imgs_u = imgs_u.to(device)   # [B_u, S, C, H, W] 3D Volume

        pred_l = sam2_model(imgs_l, prompt_masks=masks_l)
        loss_l = compute_supervised_loss(pred_l, masks_l)

        with torch.no_grad():
            feats_u = sam2_model.image_encoder(imgs_u)
            coarse_masks = pcsw_generator.forward_coarse_mask(feats_u)

            prompts = pcsw_generator.generate_prompts(coarse_masks, imgs_u.shape[-2:])

        x_w = weak_aug(imgs_u)
        x_s1 = strong_aug(x_w)
        x_s2 = strong_aug(x_w)

        f_w = sam2_model.image_encoder(x_w)
        f_s1 = sam2_model.image_encoder(x_s1)
        f_s2 = sam2_model.image_encoder(x_s2)

        f_adj1 = dfe_module(f_w, f_s1)
        f_adj2 = dfe_module(f_w, f_s2)

        with torch.no_grad():
            logits_w = sam2_model.mask_decoder(f_w, prompts)
            pseudo_label = (torch.sigmoid(logits_w) > 0.5).float()

        logits_s1 = sam2_model.mask_decoder(f_adj1, prompts)
        logits_s2 = sam2_model.mask_decoder(f_adj2, prompts)

        loss_u = 0.5 * (
            F.binary_cross_entropy_with_logits(logits_s1, pseudo_label) +
            F.binary_cross_entropy_with_logits(logits_s2, pseudo_label)
        )

        total_loss = loss_l + loss_u

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Epoch [{epoch}] Step [{step}] Loss_L: {loss_l.item():.4f} Loss_U: {loss_u.item():.4f} Total: {total_loss.item():.4f}")