

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import *
from models.homography import *
import torch.nn.init as init
from models.cspn import Affinity_Propagate
from models.propagation import Propagation
from torchvision.utils import save_image
class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, groups=4, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        # make sure that out_channels = 0 (mod groups)
        assert self.out_channels % self.groups == 0, "ERROR INPUT,CHECK AGAIN!"
        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        # Learned transformation.
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)
        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)
        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)
        # Activation here. The same with all the other conv layers.
        return nn.LeakyReLU(0.1)(out)

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)

class FeaturePyramidNet(nn.Module):
    def __init__(self, base_channels,downsample=None, reduction=0.0625, kernel_num=1):
        super(FeaturePyramidNet, self).__init__()
        #print(base_channels)
        self.conv0 = conv(3, 4*base_channels, 3, 1, 1)
        self.conv1 = odconv(4*base_channels, 4*base_channels, 3, 1, 1)
        self.conv2 = odconv(4*base_channels, 4*base_channels, 3, 1, 1)

        self.conv3 = odconv(4*base_channels, 2*base_channels, 3, 1, 1)
        self.conv4 = odconv(2*base_channels, 2*base_channels, 3, 1, 1)
        self.conv5 = odconv(2*base_channels, 2*base_channels, 3, 1, 1)

        self.conv6 = odconv(2*base_channels, base_channels, 3, 1, 1)
        self.conv7 = odconv(base_channels, base_channels, 3, 1, 1)
        self.conv8 = odconv(base_channels, base_channels, 3, 1, 1)
        self.guidance_conv= DEPTHWISECONV(16,8)
#        self.guidance_conv= conv(base_channels, 8, 3, 1, 1)
#        self.conv8 = AttentionConv(base_channels, base_channels, kernel_size=3, stride=1, groups=4)
#        self.conv9 = odconv3x3(base_channels, base_channels, reduction=reduction, kernel_num=kernel_num)
#        self.CBAM0 = CBAM(4*base_channels,16)
#        self.CBAM1 = CBAM(2*base_channels,16)
#        self.CBAM2 = CBAM(base_channels,16)  
#        self.bn2 = nn.BatchNorm2d(base_channels)
#        self.relu = nn.ReLU(inplace=True)
#        self.downsample = downsample  
    def forward(self, img, n_pyramids, use_guidance=False ):

        fp = []
        for pyramid in range(n_pyramids):
            f = self.conv0(img)
          #  f = self.CBAM0(f)

            f = self.conv1(f)
          #  f = self.CBAM0(f)

            f = self.conv2(f)
#            f = self.CBAM0(f)

            f = self.conv3(f)


            f = self.conv4(f)

            
            f = self.conv5(f)


            f = self.conv6(f)


            f = self.conv7(f)

            f = self.conv8(f)
            if use_guidance:
                guidance = self.guidance_conv(f)
                sava_image(guidance,random)
                fp.append([f, guidance ])
            else:
                fp.append(f)

            img = nn.functional.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=False).detach()

        return fp

# class BasicConv(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
#         super(BasicConv, self).__init__()
#         self.out_channels = out_planes
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
#         self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
#         self.relu = nn.ReLU() if relu else None

#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x

# class Flatten(nn.Module):
#     def forward(self, x):
#         return x.view(x.size(0), -1)

# class ChannelGate(nn.Module):
#     def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
#         super(ChannelGate, self).__init__()
#         self.gate_channels = gate_channels
#         self.mlp = nn.Sequential(
#             Flatten(),
#             nn.Linear(gate_channels, gate_channels // reduction_ratio),
#             nn.ReLU(),
#             nn.Linear(gate_channels // reduction_ratio, gate_channels)
#             )
#         self.pool_types = pool_types
#     def forward(self, x):
#         channel_att_sum = None
#         for pool_type in self.pool_types:
#             if pool_type=='avg':
#                 avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#                 channel_att_raw = self.mlp( avg_pool )
#             elif pool_type=='max':
#                 max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#                 channel_att_raw = self.mlp( max_pool )
#             elif pool_type=='lp':
#                 lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#                 channel_att_raw = self.mlp( lp_pool )
#             elif pool_type=='lse':
#                 # LSE pool only
#                 lse_pool = logsumexp_2d(x)
#                 channel_att_raw = self.mlp( lse_pool )

#             if channel_att_sum is None:
#                 channel_att_sum = channel_att_raw
#             else:
#                 channel_att_sum = channel_att_sum + channel_att_raw

#         scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
#         return x * scale

# def logsumexp_2d(tensor):
#     tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
#     s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
#     outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
#     return outputs

# class ChannelPool(nn.Module):
#     def forward(self, x):
#         return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

# class SpatialGate(nn.Module):
#     def __init__(self):
#         super(SpatialGate, self).__init__()
#         kernel_size = 7
#         self.compress = ChannelPool()
#         self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
#     def forward(self, x):
#         x_compress = self.compress(x)
#         x_out = self.spatial(x_compress)
#         scale = torch.sigmoid(x_out) # broadcasting
#         return x * scale

# class CBAM(nn.Module):
#     def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
#         super(CBAM, self).__init__()
#         self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
#         self.no_spatial=no_spatial
#         if not no_spatial:
#             self.SpatialGate = SpatialGate()
#     def forward(self, x):
#         x_out = self.ChannelGate(x)
#         if not self.no_spatial:
#             x_out = self.SpatialGate(x_out)
#         return x_out



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
        self.cspn = Affinity_Propagate(24, 3, "8sum")
        self.propagation = Propagation(in_chs=16, neighbors=8, dilation=2)
        self.prob_threshold = 0.8
    def forward(self, imgs, proj_matrices_pyramid, extrinsics_matrices, intrinsics_matrices_pyramid, init_depth_hypos, depth_mode='regression', is_training=False):
        B, V, _, H, W = imgs.shape
        n_pyramids = intrinsics_matrices_pyramid.shape[1]

        # @Note step 1. feature extraction
        ref_feature_pyramids = self.feature_pyramid(imgs[:,0], n_pyramids, use_guidance=True)
        ref_feature_pyramid = [x[0] for x in ref_feature_pyramids]
        guidance = [x[1] for x in ref_feature_pyramids]
        src_features_pyramid = []
        for i in range(1, V):
            src_features_pyramid.append(self.feature_pyramid(imgs[:,i], n_pyramids))
        del imgs


        depth = None
        depth_pyramid, confidence_pyramid = [], []
        prob_volume_pyramid, depth_hypos_pyramid, interval_pyramid = [], [], []
        
        for pyramid in range(n_pyramids-1, -1, -1):
            # print(list(range(n_pyramids-1, -1, -1)))            
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
                
                depth_hypos = self.propagation(ref_feature_pyramid[pyramid], depth_hypos)              
                #print(depth_hypos.shape)
                #print(depth_hypos[0, :, 10, 10])
                
                del depth_up
            D = depth_hypos.shape[1]


            # @Note step 2. differentiable homograph, build cost volume
            ref_feature = ref_feature_pyramid[pyramid]
            ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, D, 1, 1) # (B, F, D, h, w)
            volume_sum = ref_volume
            volume_sq_sum = ref_volume.pow_(2)

            #print(1,ref_feature.shape)
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
         
            #depth = self.cspn(guidance[pyramid], depth.unsqueeze(1), None).squeeze(1)

            prob_volume_pyramid.append(prob_volume)
            depth_hypos_pyramid.append(depth_hypos)
            interval_pyramid.append(interval)
            
            depth_pyramid.append(depth)
            confidence_pyramid.append(photometric_confidence)
        depth_pyramid, confidence_pyramid = self.confidence_check(depth_pyramid, confidence_pyramid)
        depth = self.cspn(guidance[0], depth.unsqueeze(1), None).squeeze(1)
        depth_pyramid[-1] = depth


        return {
            "depth_est_pyramid": depth_pyramid, 
            "confidence_pyramid": confidence_pyramid,
            "prob_volume_pyramid": prob_volume_pyramid,
            "depth_hypos_pyramid": depth_hypos_pyramid,
            "interval_pyramid": interval_pyramid
        }
    def net_update_temperature(self, temperature):
        for m in self.modules():
            if hasattr(m, "update_temperature"):
                m.update_temperature(temperature)
    def confidence_check(self, depth_pyramid, confidence_pyramid):
        depth_up = depth_pyramid[0]
        confidence_up = confidence_pyramid[0]
        for i in range(1, len(depth_pyramid)):
            depth_up = F.interpolate(depth_up.unsqueeze(1), size=None, scale_factor=2, mode="bilinear", align_corners=False).squeeze(1)
            confidence_up = F.interpolate(confidence_up.unsqueeze(1), size=None, scale_factor=2, mode="bilinear", align_corners=False).squeeze(1)
    
            refine_mask = ((confidence_pyramid[i]<confidence_up).int() + (confidence_up>self.prob_threshold).int()) == 2
    
            depth_pyramid[i][refine_mask] = depth_up[refine_mask]
            confidence_pyramid[i][refine_mask] = confidence_up[refine_mask]
    
            depth_up, confidence_up = depth_pyramid[i], confidence_pyramid[i]
        return depth_pyramid, confidence_pyramid
        
class DEPTHWISECONV(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DEPTHWISECONV, self).__init__()

        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

