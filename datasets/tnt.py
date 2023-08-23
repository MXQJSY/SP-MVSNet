import os
import torch
from torch.utils.data import Dataset
from datasets.datasets_io import *
from einops import rearrange
from PIL import Image
import numpy as np
import random


class TNTDataset(Dataset):
    def __init__(self, root_dir, list_file, mode, n_views, n_pyramids, n_depths, **kwargs):
        super(TNTDataset, self).__init__()
        self.root_dir = root_dir
        self.list_file = list_file
        self.mode = mode
        self.n_views = n_views
        self.n_pyramids = n_pyramids
        self.n_depths = n_depths
        
        self.total_depths = 192
        self.interval_scale = 1.06

        assert self.mode == "test"
        self.metas = self.build_metas()
        
        self.rescale = kwargs["rescale"]
        self.max_h, self.max_w = kwargs["max_h"], kwargs["max_w"]
        
        self.depth_interval_table = {
            'Family': 2.5e-3, 'Francis': 1e-2, 'Horse': 1.5e-3, 'Lighthouse': 1.5e-2, 'M60': 5e-3, 'Panther': 5e-3, 'Playground': 7e-3, 'Train': 5e-3, 
            'Auditorium': 3e-2, 'Ballroom': 2e-2, 'Courtroom': 2e-2, 'Museum': 2e-2, 'Palace': 1e-2, 'Temple': 1e-2
        }


    def build_metas(self):
        metas = []

        with open(os.path.join(self.list_file)) as f:
            scans = [line.rstrip() for line in f.readlines()]

        for scan in scans:
            with open(os.path.join(self.root_dir, scan, "pair.txt")) as f:
                num_viewpoint = int(f.readline())
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]

                    metas.append((scan, ref_view, src_views))

        print("TanksandTemples Dataset metas:", len(metas))
        return metas


    def __len__(self):
        return len(self.metas)


    def __getitem__(self, idx):
        scan, ref_view, src_views = self.metas[idx]

        view_ids = [ref_view] + src_views[:self.n_views-1]

        imgs = []
        proj_matrices_pyramid = []
        extrinsics_matrices = []
        intrinsics_matrices_pyramid = []

        for i, vid in enumerate(view_ids):
            # @Note imgs
            img_filename = os.path.join(self.root_dir, scan, "images/{:0>8}.jpg".format(vid))
            img = read_img(img_filename)

            # @Note cams
            cam_filename = os.path.join(self.root_dir, scan, "cams_{}/{:0>8}_cam.txt".format(scan.lower(), vid))
            intrinsics, extrinsics, depth_min, depth_interval = read_cam(cam_filename, self.interval_scale)
            
            if self.rescale:
                img, intrinsics = scale_img_intrinsics(img, intrinsics, self.max_w, self.max_h)
                
            imgs += [img]

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
                    "depth_max": np.array(depth_min + self.total_depths*self.depth_interval_table[scan], dtype=np.float32),
                    "n_depths": self.n_depths,
                    "depth_interval": np.array(self.depth_interval_table[scan], dtype=np.float32)
                }

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

        sample["output_dirs"] = scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"

        return sample
    