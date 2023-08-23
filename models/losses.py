
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

def regression_loss(depth_est, depth_gt, mask, epoch=0, MAX_EPOCH=12):
    mask = mask > 0.5
    
    return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')


def classification_loss(prob_volume, depth_values, interval, depth_gt, mask):
    mask = mask>0.5

    depth_gt_volume = depth_gt.unsqueeze(1).expand_as(depth_values)  # (b, d, h, w)

    gt_index_volume = (
            ((depth_values - interval / 2) <= depth_gt_volume).float() * ((depth_values + interval / 2) > depth_gt_volume).float())

    NEAR_0 = 1e-4  # Prevent overflow
    prob_volume = torch.where(prob_volume <= 0.0, torch.zeros_like(prob_volume) + NEAR_0, prob_volume)

    loss = -torch.sum(gt_index_volume * torch.log(prob_volume), dim=1)[mask].mean()

    return loss

def mvsnet_cls_loss(prob_volume, depth_gt, mask, depth_value, return_prob_map=False):
    # depth_value: B * NUM
    # get depth mask
    mask = mask>0.5
    mask_true = mask 
    valid_pixel_num = torch.sum(mask_true, dim=[1,2]) + 1e-12

    shape = depth_gt.shape

    depth_num = depth_value.shape[1]
    # depth_value_mat = depth_value.repeat(shape[1], shape[2], 1, 1).permute(2,3,0,1)
    depth_value_mat = depth_value
    gt_index_image = torch.argmin(torch.abs(depth_value_mat-depth_gt.unsqueeze(1)), dim=1)

    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1) # B, 1, H, W

    # gt index map -> gt one hot volume (B x 1 x H x W )
    gt_index_volume = torch.zeros(shape[0], depth_num, shape[1], shape[2]).type(mask_true.type()).scatter_(1, gt_index_image, 1)
    # print('shape:', gt_index_volume.shape, )
    # cross entropy image (B x D X H x W)
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume+1e-12), dim=1).squeeze(1) # B, 1, H, W
    #print('cross_entropy_image', cross_entropy_image)
    # masked cross entropy loss
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image) # valid pixel
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])

    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num) # Origin use sum : aggregate with batch
    # winner-take-all depth map
    wta_index_map = torch.argmax(prob_volume, dim=1, keepdim=True).type(torch.long)
    wta_depth_map = torch.gather(depth_value_mat, 1, wta_index_map).squeeze(1)

    if return_prob_map:
        photometric_confidence = torch.max(prob_volume, dim=1)[0] # output shape dimension B * H * W
        return masked_cross_entropy, wta_depth_map, photometric_confidence
    return masked_cross_entropy # wta_depth_map


