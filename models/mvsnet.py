
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import *
from models.homography import *


class FeaturePyramidNet(nn.Module):
    def __init__(self, base_channels):
        super(FeaturePyramidNet, self).__init__()

        self.conv0 = conv(3, 4*base_channels, 3, 1, 1)
        self.conv1 = conv(4*base_channels, 4*base_channels, 3, 1, 1)
        self.conv2 = conv(4*base_channels, 4*base_channels, 3, 1, 1)

        self.conv3 = conv(4*base_channels, 2*base_channels, 3, 1, 1)
        self.conv4 = conv(2*base_channels, 2*base_channels, 3, 1, 1)
        self.conv5 = conv(2*base_channels, 2*base_channels, 3, 1, 1)

        self.conv6 = conv(2*base_channels, base_channels, 3, 1, 1)
        self.conv7 = conv(base_channels, base_channels, 3, 1, 1)
        self.conv8 = conv(base_channels, base_channels, 3, 1, 1)

    def forward(self, img, n_pyramids):

        fp = []
        for pyramid in range(n_pyramids):
            f = self.conv2(self.conv1(self.conv0(img)))
            f = self.conv5(self.conv4(self.conv3(f)))
            f = self.conv8(self.conv7(self.conv6(f)))
            fp.append(f)

            img = nn.functional.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=False).detach()

        return fp


class CostRegNet(nn.Module):
    def __init__(self, base_channels):
        super(CostRegNet, self).__init__()

        self.conv0 = ConvBnReLU3D(base_channels, base_channels, 3, 1, 1)
        self.conv1 = ConvBnReLU3D(base_channels, base_channels, 3, 1, 1)

        self.conv2 = ConvBnReLU3D(base_channels, 2*base_channels, 3, 2, 1)
        self.conv3 = ConvBnReLU3D(2*base_channels, 2*base_channels, 3, 1, 1)
        self.conv4 = ConvBnReLU3D(2*base_channels, 2*base_channels, 3, 1, 1)

        self.conv5 = ConvBnReLU3D(2*base_channels, 4*base_channels, 3, 1, 1)
        self.conv6 = ConvBnReLU3D(4*base_channels, 4*base_channels, 3, 1, 1)
        self.conv7 = ConvBnReLU3D(4*base_channels, 4*base_channels, 3, 1, 1)

        self.conv8 = nn.Sequential(
            nn.ConvTranspose3d(4*base_channels, 2*base_channels, kernel_size=3, stride=1, padding=1, output_padding=0, bias=False),
            nn.BatchNorm3d(2*base_channels),
            nn.ReLU(inplace=True)
        )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(2*base_channels, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )

        self.prob = nn.Conv3d(base_channels, 1, 3, 1, 1)

    def forward(self, x):
        conv0 = self.conv1(self.conv0(x))
        conv4 = self.conv4(self.conv3(self.conv2(conv0)))
        conv7 = self.conv7(self.conv6(self.conv5(conv4)))

        conv8 = conv4 + self.conv8(conv7)
        del conv4, conv7
        conv9 = conv0 + self.conv9(conv8)
        del conv0, conv8

        prob = self.prob(conv9).squeeze(1)
        del conv9

        return prob


class MVSNet(nn.Module):
    def __init__(self, which_dataset, base_channels=16, refine=False):
        super(MVSNet, self).__init__()
        self.which_dataset = which_dataset

        self.feature_pyramid = FeaturePyramidNet(base_channels)
        self.cost_regularization = CostRegNet(base_channels)
        
        
    def forward(self, imgs, proj_matrices_pyramid, extrinsics_matrices, intrinsics_matrices_pyramid, init_depth_hypos, depth_mode='regression', is_training=False):
        B, V, _, H, W = imgs.shape
        n_pyramids = intrinsics_matrices_pyramid.shape[1]

        # @Note step 1. feature extraction
        ref_feature_pyramid = self.feature_pyramid(imgs[:,0], n_pyramids)
        src_features_pyramid = []
        for i in range(1, V):
            src_features_pyramid.append(self.feature_pyramid(imgs[:,i], n_pyramids))
        del imgs


        depth = None
        depth_pyramid, confidence_pyramid = [], []
        prob_volume_pyramid, depth_hypos_pyramid, interval_pyramid = [], [], []
        
        for pyramid in range(n_pyramids-1, -1, -1):
            coarse_stage_flag = True if pyramid == n_pyramids-1 else False

            # @Note camera parameters
            ref_extrinsics, src_extrinsics = extrinsics_matrices[:, 0], extrinsics_matrices[:, 1:]
            intrinsics_matrices = intrinsics_matrices_pyramid[:, pyramid]
            ref_intrinsics, src_intrinsics = intrinsics_matrices[:, 0], intrinsics_matrices[:, 1:]

            proj_matrices = proj_matrices_pyramid[:, pyramid]
            ref_proj, src_projs = proj_matrices[:, 0], proj_matrices[:, 1:]


            # @Note depth plane hypothetical
            if coarse_stage_flag:
                depth_hypos = calInitDepthHypos(init_depth_hypos, ref_feature_pyramid[pyramid].shape)
                interval = init_depth_hypos["depth_interval"][0]
            else:
                if is_training:
                    if self.which_dataset=="dtu":
                        n_depths_pyramid = [8]
                        ratio_pyramid = [2.5]
                    elif self.which_dataset=="blendedmvs":
                        n_depths_pyramid = [8, 8, 8]
                        ratio_pyramid = [0.25, 0.5, 1]
                else:
                    if self.which_dataset == "dtu":
                        n_depths_pyramid = [8, 8, 8, 8]
                        ratio_pyramid = [0.25, 0.5, 1, 1]
                    elif self.which_dataset == "tnt":
                        n_depths_pyramid = [8, 8, 16, 32]
                        ratio_pyramid = [1, 2, 3, 4]
                    elif self.which_dataset == "blendedmvs":
                        n_depths_pyramid = [8, 8, 8]
                        ratio_pyramid = [0.25, 0.5, 1]

                depth_up = F.interpolate(depth.unsqueeze(1), size=None, scale_factor=2, mode="bilinear", align_corners=False).squeeze(1)
                depth_hypos = calDeptyHypos(depth_up, init_depth_hypos, n_depths_pyramid[pyramid], ratio_pyramid[pyramid])  
                interval = init_depth_hypos["depth_interval"][0] * ratio_pyramid[pyramid]     
                del depth_up
            D = depth_hypos.shape[1]


            # @Note step 2. differentiable homograph, build cost volume
            ref_feature = ref_feature_pyramid[pyramid]
            ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, D, 1, 1) # (B, F, D, h, w)
            volume_sum = ref_volume
            volume_sq_sum = ref_volume.pow_(2)

            for src_view in range(1, V):
                src_feat = src_features_pyramid[src_view-1][pyramid]
                src_in = src_intrinsics[:,src_view-1]
                src_ex = src_extrinsics[:,src_view-1]

                src_proj = src_projs[:,src_view-1]

                # warped_volume = homo_warping(src_feat, ref_intrinsics, src_in, ref_extrinsics, src_ex, depth_hypos)     # for pytorch<=1.2.0
                warped_volume = homo_warping_stable(src_feat, src_proj, ref_proj, depth_hypos)        # for high version pytorch(e.g. 1.4.0, 1.8.0)

                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2

                del warped_volume
            # aggregate multiple feature volumes by variance
            cost_volume = volume_sq_sum.div_(V).sub_(volume_sum.div_(V).pow_(2))
            del volume_sq_sum, volume_sum

            # @Note step 3. cost volume regularization
            cost_reg = self.cost_regularization(cost_volume)
            del cost_volume


            if depth_mode == 'regression':
                prob_volume = F.softmax(cost_reg, dim=1) # (B, D, h, w)
                depth = depth_regression(prob_volume, depth_hypos)
                with torch.no_grad():
                    prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1) # (B, D, h, w)
                    depth_index = depth_regression(prob_volume, torch.arange(D, device=prob_volume.device, dtype=prob_volume.dtype)).long() # (B, h, w)
                    depth_index = depth_index.clamp(min=0, max=D-1)
                    photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1) # (B, h, w)
            elif depth_mode == 'classification':
                prob_volume = F.softmax(cost_reg, dim=1) # (B, D, h, w)
                depth = winner_take_all(prob_volume, depth_hypos)
                photometric_confidence, _ = torch.max(prob_volume, dim=1)
                
            prob_volume_pyramid.append(prob_volume)
            depth_hypos_pyramid.append(depth_hypos)
            interval_pyramid.append(interval)
            
            depth_pyramid.append(depth)
            confidence_pyramid.append(photometric_confidence)


        return {
            "depth_est_pyramid": depth_pyramid, 
            "confidence_pyramid": confidence_pyramid,
            "prob_volume_pyramid": prob_volume_pyramid,
            "depth_hypos_pyramid": depth_hypos_pyramid,
            "interval_pyramid": interval_pyramid
        }