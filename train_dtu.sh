#!/usr/bin/env bash
source /home/yhh/bianjilong/ssd/cvp1/data_path.sh


THISNAME="odmvs_hybrid_57yz"
THISGPU=0       # 5  3,4


# Train
echo "<<<<<<<<<< start train"
#conda activate ghc_ourmvsnet_stable
CUDA_VISIBLE_DEVICES=$THISGPU python train.py \
    --mode="train" \
    --exp_name=$THISNAME \
    --ckptdir="./checkpoints/"$THISNAME \
    --logdir="./checkpoints/"$THISNAME \
    --num_epochs=16 \
    --batch_size=6 --lr=0.0005 \
    --num_gpus=1 \
    \
    --root_dir=$DTU_TRAIN_ROOT \
    --list_file="datasets/lists/dtu/train.txt" \
    --n_depths=48 \
    --n_views=5 \
    --n_pyramids=2 \
    \
    --root_eval_dir=$DTU_TEST_ROOT \
    --list_eval_file="datasets/lists/dtu/test.txt" \
    --n_depths_eval=48 \
    --n_views_eval=7 \
    --n_pyramids_eval=5 \
    \
    --depth_mode="classification" \

    # --resume
