"""
2022/03/12, doubleZ, PKU
DTU Dataset.
"""

import os
from torch.utils.data import Dataset
from datasets.datasets_io import *
from einops import rearrange
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms


def motion_blur(img: np.ndarray, max_kernel_size=3):
    # Either vertial, hozirontal or diagonal blur
    mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
    ksize = np.random.randint(0, (max_kernel_size + 1) / 2) * 2 + 1  # make sure is odd
    center = int((ksize - 1) / 2)
    kernel = np.zeros((ksize, ksize))
    if mode == 'h':
        kernel[center, :] = 1.
    elif mode == 'v':
        kernel[:, center] = 1.
    elif mode == 'diag_down':
        kernel = np.eye(ksize)
    elif mode == 'diag_up':
        kernel = np.flip(np.eye(ksize), 0)
    var = ksize * ksize / 16.
    grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
    gaussian = np.exp(-(np.square(grid - center) + np.square(grid.T - center)) / (2. * var))
    kernel *= gaussian
    kernel /= np.sum(kernel)
    img = cv2.filter2D(img, -1, kernel)
    return img


class BlendedMVSDataset(Dataset):
    def __init__(self, root_dir, list_file, mode, n_views, n_pyramids, n_depths):
        super(BlendedMVSDataset, self).__init__()
        self.root_dir = root_dir
        self.list_file = list_file
        self.mode = mode
        self.n_views = n_views
        self.n_pyramids = n_pyramids
        self.n_depths = n_depths

        self.total_depths = 192
        self.interval_scale = 1.06

        assert self.mode in ["train", "val"]

        self.metas = self.build_metas()
        self.transform = transforms.ColorJitter(brightness=0.25, contrast=(0.3, 1.5))


    def build_metas(self):
        metas = []

        with open(os.path.join(self.list_file)) as f:
            scans = [line.rstrip() for line in f.readlines()]

        for scan in scans:
            with open(os.path.join(self.root_dir, scan, "cams/pair.txt")) as f:
                num_viewpoint = int(f.readline())
                for view in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    line = f.readline().rstrip().split()
                    n_views_valid = int(line[0])        # valid views
                    if n_views_valid < self.n_views:    # skip no enough valid views
                        continue
                    if self.mode=="val" and view>10:    # only val 10 views per scan
                        continue
                    src_views = [int(x) for x in line[1::2]]
                    metas.append((scan, ref_view, src_views))

        print("BlendedMVS Dataset in", self.mode, "mode metas:", len(metas))
        return metas


    def __len__(self):
        return len(self.metas)


    def read_img_blended(self, filename):
        img = Image.open(filename)
        if self.mode == "train":
            img = self.transform(img)
            img = motion_blur(np.array(img, dtype=np.float32))

        np_img = np.array(img, dtype=np.float32) / 255.0
        return np_img


    def __getitem__(self, idx):
        scan, ref_view, src_views = self.metas[idx]

        view_ids = [ref_view] + src_views[:self.n_views-1]

        imgs = []
        proj_matrices_pyramid = []
        extrinsics_matrices = []
        intrinsics_matrices_pyramid = []

        for i, vid in enumerate(view_ids):

            # @Note imgs
            img_filename = os.path.join(self.root_dir, scan, "blended_images/{:0>8}.jpg".format(vid))
            imgs += [self.read_img_blended(img_filename)]

            # @Note cams
            proj_mat_filename = os.path.join(self.root_dir, scan, "cams/{:0>8}_cam.txt".format(vid))
            intrinsics, extrinsics, depth_min, depth_interval = read_cam(proj_mat_filename, self.interval_scale)    

            proj_matrices = []
            intrinsics_matrices = []
            for _ in range(self.n_pyramids):          
                proj_mat = extrinsics.copy()
                proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
                proj_matrices += [proj_mat]
                intrinsics_matrices += [intrinsics.copy()]
                intrinsics[:2, :] /= 2
            proj_matrices_pyramid += [proj_matrices]
            extrinsics_matrices += [extrinsics]
            intrinsics_matrices_pyramid += [intrinsics_matrices]

            if i == 0:
                # @Note depth values
                init_depth_hypos = {
                    "depth_min": np.array(depth_min, dtype=np.float32),
                    "depth_max": np.array(depth_min+self.total_depths*depth_interval, dtype=np.float32),
                    "n_depths": self.n_depths,
                    "depth_interval": np.array(depth_interval, dtype=np.float32),
                    "init_depth_hypos": np.arange(depth_min, depth_interval * (self.n_depths - 0.5) + depth_min, depth_interval, dtype=np.float32)
                }

                # @Note depth_gt, mask
                depth_filename = os.path.join(self.root_dir, scan, "rendered_depth_maps/{:0>8}.pfm".format(vid))

                depth = read_depth(depth_filename)
                depth_frame_size = (depth.shape[0], depth.shape[1])
                frame = np.zeros(depth_frame_size)
                frame[:,:] = depth

                depth_pyramid = [frame]

                down_depth = Image.fromarray(depth_pyramid[0])
                origin_size = np.array(down_depth.size).astype(int)
                for pyramid in range(1, self.n_pyramids):
                    fresh_size = (origin_size / (2**pyramid)).astype(int)
                    down_depth = np.array(down_depth.resize((fresh_size), Image.BICUBIC))
                    down_depth_frame_size = (down_depth.shape[0], down_depth.shape[1])
                    frame = np.zeros(down_depth_frame_size)
                    frame[:,:] = down_depth
                    down_depth = Image.fromarray(down_depth)

                    depth_pyramid = [frame] + depth_pyramid

                mask_pyramid = []
                for depth in depth_pyramid:
                    m = np.ones(depth.shape, np.bool)
                    m[depth<init_depth_hypos["depth_min"]] = 0
                    mask_pyramid.append(m)


        imgs = rearrange(np.stack(imgs), 'V h w C -> V C h w')
        proj_matrices_pyramid = rearrange(np.stack(proj_matrices_pyramid), 'V pyramid a b -> pyramid V a b')
        extrinsics_matrices = np.stack(extrinsics_matrices)
        intrinsics_matrices_pyramid = rearrange(np.stack(intrinsics_matrices_pyramid), 'V pyramid a b -> pyramid V a b')

        sample = {
            "imgs": imgs,
            "proj_matrices_pyramid": proj_matrices_pyramid,
            "camera_parameter": {
                "extrinsics_matrices": extrinsics_matrices,
                "intrinsics_matrices_pyramid": intrinsics_matrices_pyramid
            },
            "init_depth_hypos": init_depth_hypos,
            "depth_gt_pyramid": depth_pyramid,
            "mask_pyramid": mask_pyramid,
        }

        return sample