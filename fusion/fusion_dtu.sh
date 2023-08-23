#!/usr/bin/env bash
source /hy-tmp/ODN_CVPMVSNet_PKU/scripts/data_path.sh
THISNAME="aa9"
echo "<<<<<<<<<< start UCS fusion"
# conda activate ourmvsnet_stable
CUDA_VISIBLE_DEVICES=0 python fusion_ucs.py \
     --root_path=$DTU_TEST_ROOT \
     --depth_path="/hy-tmp/ODN_CVPMVSNet_PKU/scripts/outputs/dtu/"$THISNAME \
     --ply_path="/hy-tmp/ODN_CVPMVSNet_PKU/scripts/outputs/dtu/"$THISNAME"/ucs_fusion_plys/" \
     --data_list="/hy-tmp/ODN_CVPMVSNet_PKU/scripts/datasets/lists/dtu/test.txt" \
     --prob_thresh=0.3 \
     --dist_thresh=0.25 \
     --num_consist=3 \
     --device="cuda"