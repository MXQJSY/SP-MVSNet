# Our_CVPMVSNet_baseline

## Setup
1. Download CVP-MVSNet Dataset[train | test]

2. Setup conda environment

   - env for training & testing

      ```bash
      conda create -n dbzmvsnet python=3.6
      conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
      conda install -c https://conda.anaconda.org/menpo opencv
      conda install scipy scikit-learn matplotlib Pillow tqdm 
      conda install -c conda-forge protobuf==3.8.0 tensorboardX==1.8 tensorboard==1.14.0 absl-py grpcio
      conda install einops
      ```

   - env for fusion `conda install -n fusibile python=2.7` and the following packages to install

      ```
      opencv-python==4.1.1.26
      scipy==1.2.2
      tensorflow==1.14.0
      matplotlib==2.2.4
      ```

3. Download, compile and link the `fusibile`

    ```bash
    git clone https://github.com/YoYo000/fusibile.git
    cd fusion/fusibile
    mkdir build && cd build
    cmake ..
    make
    
    ln -s /path/to/fusibile/build/fusibile /path/to/fusion/gipuma/fusibile
    ```

    > you can find some useful information [here](https://zhuanlan.zhihu.com/p/460212787) if you have problems for installing the fusibile

4. Make `checkpoints/`, `logs/` in the root dir

5. Modify the related paths mentioned in *Line4~9* in `alembic.sh`

6. Run `alembic.sh` for `train -> test -> fusion`, and you will get `.ply` point cloud files finally if all settings are done right.

6. Use `outputs/visual.ipynb` to visualize the depth map.

