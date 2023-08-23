#!/usr/bin/env bash
source scripts/data_path.sh


# THISNAME="odmvs_1117"
THISGPU=2

# Fusion(Gipuma)
echo "<<<<<<<<<< start fusion"
conda activate ghc_fusibile
CUDA_VISIBLE_DEVICES=$THISGPU python2 fusion/dtu/fusion_gipuma.py \
    --root_dir=$DTU_TEST_ROOT \
    --list_file="datasets/lists/dtu/test.txt" \
    --fusibile_exe_path="fusion/fusibile" \
    --depth_folder="./outputs/dtu/"$THISNAME \
    --out_folder="fusibile_fused" \
    --ply_path="./outputs/dtu/"$THISNAME"/gipuma_fusion_plys/" \
    --quan_path="./outputs/dtu/"$THISNAME"/gipuma_quantitative/" \
    --prob_threshold=0.3 \
    --disp_threshold=0.25 \
    --num_consistent=3 \
    --downsample_factor=1 \
    
    # --quan_root_dir=$DTU_QUANTITATIVE_ROOT \
    # --matlab_quantitative

# --prob_threshold=0.8 --disp_threshold=0.13 for 'regression' 0.3 0.25
wait
bash scripts/matlab_quan_dtu.sh

# # ==========================================================
# echo "<<<<<<<<<< start UCS fusion"
# conda activate ourmvsnet_stable
# CUDA_VISIBLE_DEVICES=$THISGPU python fusion/dtu/fusion_ucs.py \
#     --root_path=$DTU_TEST_ROOT \
#     --depth_path="./outputs/dtu/"$THISNAME \
#     --ply_path="./outputs/dtu/"$THISNAME"/ucs_fusion_plys/" \
#     --quan_path="./outputs/dtu/"$THISNAME"/ucs_quantitative/" \
#     --data_list="datasets/lists/dtu/test.txt" \
#     --prob_thresh=0.3 \
#     --dist_thresh=0.25 \
#     --num_consist=3 \
#     --device="cuda"