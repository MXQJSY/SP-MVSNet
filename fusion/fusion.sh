# Shell script for running fusibile depth fusion
# by: Jiayu Yang
# date: 2019-11-05

DTU_TEST_ROOT="/home/yhh/bianjilong/ssd/datasets/dtu-test-1200"
DEPTH_FOLDER="/home/yhh/bianjilong/ssd/CVP2-A5000-cas/outputs/dtu/odmvs_hybrid_2/"
OUT_FOLDER="/home/yhh/bianjilong/ssd/CVP2-A5000-cas/outputs/dtu/odmvs_hybrid_2/"
FUSIBILE_EXE_PATH="/home/yhh/bianjilong/ssd/fusibile/build/fusibile"

python2 depthfusion.py \
--dtu_test_root=$DTU_TEST_ROOT \
--depth_folder=$DEPTH_FOLDER \
--out_folder=$OUT_FOLDER \
--fusibile_exe_path=$FUSIBILE_EXE_PATH \
--prob_threshold=0.1 \
--disp_threshold=0.25 \
--num_consistent=3
