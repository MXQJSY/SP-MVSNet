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
import random

class DTUDataset(Dataset):
    def __init__(self, root_dir, list_file, mode, n_views, n_pyramids, n_depths, **kwargs):
        super(DTUDataset, self).__init__()
        self.root_dir = root_dir
        self.list_file = list_file
        self.mode = mode
        self.n_views = n_views
        self.n_pyramids = n_pyramids
        self.n_depths = n_depths

        self.total_depths = 192
        self.interval_scale = 1.06

        assert self.mode in ["train", "val", "test"]

        self.metas = self.build_metas()


    def build_metas(self):
        metas = []

        # e.g. datasets/list/dtu/train.txt
        with open(os.path.join(self.list_file)) as f:
            scans = [line.rstrip() for line in f.readlines()]

        pair_file = "Cameras/pair.txt"
        eval_scan_table = {
            'scan1': 1, 'scan4': 1, 'scan10': 1, 'scan12': 1, 'scan34': 1, 'scan110': 1, 'scan114': 1, 'scan118': 2,
            'scan9': 2, 'scan13': 1, 'scan15': 2, 'scan23': 2, 'scan24': 2, 'scan32': 4,
            'scan11': 10, 'scan29': 30, 'scan33': 10, 'scan75': 10,
            'scan48': 15, 'scan49': 15, 'scan62': 15, 'scan77': 30
        }

        for scan in scans:

            with open(os.path.join(self.root_dir, pair_file)) as f:
                num_viewpoint = int(f.readline())

                if self.mode == "val":
                    eval_num = eval_scan_table[scan]
                    random.seed(2022)
                    eval_choices = random.choices([view for view in range(num_viewpoint)], k=eval_num)

                # viewpoints (49)
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]

                    if self.mode == "train":
                        # light conditions 0-6
                        for light_idx in range(7):
                            metas.append((scan, light_idx, ref_view, src_views))
                    elif self.mode == "test":
                        metas.append((scan, 3, ref_view, src_views))
                    elif self.mode == "val":
                        if ref_view in eval_choices:
                            metas.append((scan, 3, ref_view, src_views))

        print("DTU Dataset in", self.mode, "mode metas:", len(metas))
        return metas


    def __len__(self):
        return len(self.metas)


    def __getitem__(self, idx):
        scan, light_idx, ref_view, src_views = self.metas[idx]
        view_ids = [ref_view] + src_views[:self.n_views-1]

        imgs = []
        proj_matrices_pyramid = []
        extrinsics_matrices = []
        intrinsics_matrices_pyramid = []

        for i, vid in enumerate(view_ids):

            # @Note imgs(1~49)
            img_filename = os.path.join(self.root_dir, 'Rectified/{}{}/rect_{:0>3}_{}_r5000.png'.format(scan, "" if self.mode in ['test', "val"] else "_train", vid + 1, light_idx))
            imgs += [read_img(img_filename)]
             
            # @Note cams
            proj_mat_filename = os.path.join(self.root_dir, 'Cameras/{:0>8}_cam.txt').format(vid)
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
                    "depth_interval": np.array(depth_interval, dtype=np.float32)
                }

                # @Note depth_gt, mask
                if self.mode == "train":
                    depth_filename = os.path.join(self.root_dir, 'Depths/{}_train/depth_map_{:0>4}.pfm'.format(scan, vid))
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

                        depth_pyramid = [frame] + depth_pyramid

                    mask_pyramid = []
                    for depth in depth_pyramid:
                        m = np.ones(depth.shape, np.float32)
                        m[depth>init_depth_hypos["depth_max"]] = 0
                        m[depth<init_depth_hypos["depth_min"]] = 0
                        mask_pyramid.append(m)
                elif self.mode == "val":
                    depth_filename = os.path.join(self.root_dir, 'Depths/{}/depth_map_{:0>4}.pfm'.format(scan, vid))
                    mask_filename = os.path.join(self.root_dir, 'Depths/{}/depth_visual_{:0>4}.png'.format(scan, vid))

                    depth_pyramid = [read_depth(depth_filename)]
                    mask_pyramid = [read_mask(mask_filename)]
                    depth, mask = depth_pyramid[0], mask_pyramid[0]
                    h, w = depth.shape
                    for pyramid in range(1, self.n_pyramids):
                        depth_pyramid = [cv2.resize(depth, (w//2**pyramid, h//2**pyramid), interpolation=cv2.INTER_LINEAR)] + depth_pyramid
                        mask_pyramid = [cv2.resize(mask, (w//2**pyramid, h//2**pyramid), interpolation=cv2.INTER_LINEAR)] + mask_pyramid


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
        }

        if self.mode in ["train", "val"]:
            sample["depth_gt_pyramid"] = depth_pyramid
            sample["mask_pyramid"] = mask_pyramid
        elif self.mode == "test":
            sample["output_dirs"] = scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"

        return sample