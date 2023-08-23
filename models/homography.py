
import torch
import torch.nn.functional as F
from kornia.utils import create_meshgrid


def homo_warping(src_feature, ref_in, src_in, ref_ex, src_ex, depth_hypos):
    batch, channels = src_feature.shape[0], src_feature.shape[1]
    num_depth = depth_hypos.shape[1]
    height, width = src_feature.shape[2], src_feature.shape[3]

    with torch.no_grad():
        src_proj = torch.matmul(src_in,src_ex[:,0:3,:])
        ref_proj = torch.matmul(ref_in,ref_ex[:,0:3,:])
        last = torch.tensor([[[0,0,0,1.0]]]).repeat(len(src_in),1,1).cuda()
        src_proj = torch.cat((src_proj,last),1)
        ref_proj = torch.cat((ref_proj,last),1)

        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_feature.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_feature.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]

        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_hypos.view(batch, 1, num_depth, -1)
            
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        # proj_xyz[:, 2:3][proj_xyz[:, 2:3] == 0] += 0.00001  # NAN BUG, not on dtu, but on blendedmvs
        
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_feature, grid.view(batch, num_depth * height, width, 2), mode='bilinear', padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea


def homo_warping_stable(src_feat, src_proj, ref_proj, depth_values):
    B, C, H, W = src_feat.shape
    D = depth_values.shape[1]
    device = src_feat.device
    dtype = src_feat.dtype

    with torch.no_grad():
        transform = src_proj @ torch.inverse(ref_proj)
        R = transform[:, :3, :3]
        T = transform[:, :3, 3:]
        # create grid from the ref frame
        ref_grid = create_meshgrid(H, W, normalized_coordinates=False)
        ref_grid = ref_grid.to(device).to(dtype)
        ref_grid = ref_grid.permute(0, 3, 1, 2)
        ref_grid = ref_grid.reshape(1, 2, H*W)
        ref_grid = ref_grid.expand(B, -1, -1)
        ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[:,:1])), 1)
        ref_grid_d = ref_grid.unsqueeze(2) * depth_values.view(B, 1, D, -1)
        ref_grid_d = ref_grid_d.view(B, 3, D*H*W)
        src_grid_d = R @ ref_grid_d + T
        del ref_grid_d, ref_grid, transform, R, T

        # project negative depth pixels to somewhere outside the image
        negative_depth_mask = src_grid_d[:, 2:] <= 1e-7
        src_grid_d[:, 0:1][negative_depth_mask] = W
        src_grid_d[:, 1:2][negative_depth_mask] = H
        src_grid_d[:, 2:3][negative_depth_mask] = 1

        src_grid = src_grid_d[:, :2] / src_grid_d[:, -1:]
        del src_grid_d
        src_grid[:, 0] = src_grid[:, 0]/((W - 1) / 2) - 1
        src_grid[:, 1] = src_grid[:, 1]/((H - 1) / 2) - 1
        src_grid = src_grid.permute(0, 2, 1)
        
    src_grid = src_grid.view(B, D, H*W, 2)
    warped_src_feat = F.grid_sample(src_feat, src_grid, mode='bilinear', padding_mode='zeros', align_corners=True)  # remove align_corners in low version pytorch

    warped_src_feat = warped_src_feat.view(B, C, D, H, W)

    return warped_src_feat