#!/usr/bin/env bash
source /home/yhh/bianjilong/ssd/cvp1/data_path.sh

THISNAME="odmvs_hybrid_7"
BESTEPOCH="02" #"03"  # 15  05
THISGPU=1       # 5  3,4


# Test
echo "<<<<<<<<<< start eval"
#conda activate ghc_ourmvsnet_stable
CUDA_VISIBLE_DEVICES=$THISGPU python test.py \
    --mode="test" \
    --ckpt_path="./checkpoints/"$THISNAME"/model_"$BESTEPOCH".ckpt" \
    --outdir="./outputs/dtu/"$THISNAME \
    --batch_size=1 \
    \
    --root_dir=$DTU_TEST_ROOT \
    --list_file="datasets/lists/dtu/test.txt" \
    --n_depths=48 \
    --n_views=7 \
    --n_pyramids=5 \
    \
    --depth_mode="classification" \

wait
#conda activate cvp
#bash /hy-tmp/fusion/fusion.sh
