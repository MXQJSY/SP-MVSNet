import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnReLU3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        pad: int = 1,
        dilation: int = 1,
    ) -> None:

        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return F.relu(self.bn(self.conv(x)), inplace=True)


def get_grid(
        grid_type: int,
        batch: int,
        height: int,
        width: int,
        offset: torch.Tensor,
        device: torch.device,
        propagate_neighbors: int,
        evaluate_neighbors: int,
        dilation: int,
) -> torch.Tensor:
    """Compute the offset for adaptive propagation or spatial cost aggregation in adaptive evaluation

    Args:
        grid_type: type of grid - propagation (1) or evaluation (2)
        batch: batch size
        height: grid height
        width: grid width
        offset: grid offset
        device: device on which to place tensor

    Returns:
        generated grid: in the shape of [batch, propagate_neighbors*H, W, 2]
    """
    grid_types = {"propagation": 1, "evaluation": 2}

    if grid_type == grid_types["propagation"]:
        if propagate_neighbors == 4:  # if 4 neighbors to be sampled in propagation
            original_offset = [[-dilation, 0], [0, -dilation], [0, dilation], [dilation, 0]]
        elif propagate_neighbors == 8:  # if 8 neighbors to be sampled in propagation
            original_offset = [
                [-dilation, -dilation],
                [-dilation, 0],
                [-dilation, dilation],
                [0, -dilation],
                [0, dilation],
                [dilation, -dilation],
                [dilation, 0],
                [dilation, dilation],
            ]
        elif propagate_neighbors == 16:  # if 16 neighbors to be sampled in propagation
            original_offset = [
                [-dilation, -dilation],
                [-dilation, 0],
                [-dilation, dilation],
                [0, -dilation],
                [0, dilation],
                [dilation, -dilation],
                [dilation, 0],
                [dilation, dilation],
            ]
            for i in range(len(original_offset)):
                offset_x, offset_y = original_offset[i]
                original_offset.append([2 * offset_x, 2 * offset_y])
        else:
            raise NotImplementedError
    elif grid_type == grid_types["evaluation"]:
        dilation = dilation - 1  # dilation of evaluation is a little smaller than propagation
        if evaluate_neighbors == 9:  # if 9 neighbors to be sampled in evaluation
            original_offset = [
                [-dilation, -dilation],
                [-dilation, 0],
                [-dilation, dilation],
                [0, -dilation],
                [0, 0],
                [0, dilation],
                [dilation, -dilation],
                [dilation, 0],
                [dilation, dilation],
            ]
        elif evaluate_neighbors == 17:  # if 17 neighbors to be sampled in evaluation
            original_offset = [
                [-dilation, -dilation],
                [-dilation, 0],
                [-dilation, dilation],
                [0, -dilation],
                [0, 0],
                [0, dilation],
                [dilation, -dilation],
                [dilation, 0],
                [dilation, dilation],
            ]
            for i in range(len(original_offset)):
                offset_x, offset_y = original_offset[i]
                if offset_x != 0 or offset_y != 0:
                    original_offset.append([2 * offset_x, 2 * offset_y])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    with torch.no_grad():
        y_grid, x_grid = torch.meshgrid(
            [
                torch.arange(0, height, dtype=torch.float32, device=device),
                torch.arange(0, width, dtype=torch.float32, device=device),
            ]
        )
        y_grid, x_grid = y_grid.contiguous().view(height * width), x_grid.contiguous().view(height * width)
        xy = torch.stack((x_grid, y_grid))  # [2, H*W]
        xy = torch.unsqueeze(xy, 0).repeat(batch, 1, 1)  # [B, 2, H*W]

    xy_list = []
    for i in range(len(original_offset)):
        original_offset_y, original_offset_x = original_offset[i]
        offset_x = original_offset_x + offset[:, 2 * i, :].unsqueeze(1)
        offset_y = original_offset_y + offset[:, 2 * i + 1, :].unsqueeze(1)
        xy_list.append((xy + torch.cat((offset_x, offset_y), dim=1)).unsqueeze(2))

    xy = torch.cat(xy_list, dim=2)  # [B, 2, 9, H*W]

    del xy_list
    del x_grid
    del y_grid

    x_normalized = xy[:, 0, :, :] / ((width - 1) / 2) - 1
    y_normalized = xy[:, 1, :, :] / ((height - 1) / 2) - 1
    del xy
    grid = torch.stack((x_normalized, y_normalized), dim=3)  # [B, 9, H*W, 2]
    del x_normalized
    del y_normalized
    return grid.view(batch, len(original_offset) * height, width, 2)


class ConvBNReLU(nn.Module):
    def __init__(self,
                 inchs: int,
                 outchs: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 groups: int = 1,
                 bias: bool = False,
                 ) -> None:
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(inchs, outchs, kernel_size, stride, (kernel_size-1)//2, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(outchs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,
                x: torch.Tensor,
                ) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


"""
    ap
"""
class Propagation(nn.Module):
    def __init__(self,
                 in_chs,
                 neighbors,
                 dilation,
                 ):
        super(Propagation, self).__init__()

        self.neighbors = neighbors
        self.dilation = dilation
        self.grid_type = {"propagation": 1, "evaluation": 2}

        self.propa_conv = nn.Conv2d(
            in_channels=in_chs,
            out_channels=max(2 * neighbors, 1),
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=True,
        )
        nn.init.constant_(self.propa_conv.weight, 0.0)
        nn.init.constant_(self.propa_conv.bias, 0.0)

    def forward(self,
                ref_feature: torch.Tensor,
                depth_hypos: torch.Tensor,
                ) -> torch.Tensor:  # [batch, num_depth+num_neighbors, height, width]
        B, C, H, W = ref_feature.shape
        device = ref_feature.device

        propa_offset = self.propa_conv(ref_feature).view(B, 2 * self.neighbors, H * W)
        propa_grid = get_grid(self.grid_type["propagation"], B, H, W, propa_offset, device, self.neighbors, 0,
                                       self.dilation)

        batch, num_depth, height, width = depth_hypos.size()
        num_neighbors = propa_grid.size()[1] // height

        # num_depth//2 is nearest depth map
        propagate_depth_hypos = F.grid_sample(
            depth_hypos[:, num_depth // 2, :, :].unsqueeze(1), propa_grid,
            mode="bilinear", padding_mode="border", align_corners=False
        ).view(batch, num_neighbors, height, width)

        return torch.sort(torch.cat((depth_hypos, propagate_depth_hypos), dim=1), dim=1)[0]


if __name__ == "__main__":
    ref_feature = torch.randn(2, 64, 32, 32)
    depth_hypos = torch.randn(2, 8, 32, 32)

    ap = Propagation(in_chs=64, neighbors=8, dilation=2)
    ap_depth_hypos = ap(ref_feature, depth_hypos)
    print(ap_depth_hypos.shape)  #(2, 16, 32, 32)

    for b in range(2):
        for i in range(32):
            for j in range(32):
                print(ap_depth_hypos[:,:,i,j])
