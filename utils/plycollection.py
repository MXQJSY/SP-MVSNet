"""
2022/03/12, doubleZ, PKU
Point cloud files collection.
"""

import os
import shutil
import argparse

def collect(thisname, method):
    base = os.path.join('outputs/', thisname, 'fusibile_fused')
    model_name = 'final3d_model.ply'
    target = os.path.join('outputs/', thisname, method + '_results')
    info = ''

    if not os.path.exists(target + info):
        os.makedirs(target + info)

    for dir in os.listdir(base):
        scan = os.listdir(os.path.join(base, dir))
        index = dir[4:]
        model_dir = [item for item in scan if item.startswith("consistencyCheck")][0]

        old = os.path.join(base, dir, model_dir, model_name)
        fresh = os.path.join(target, info, method) + index.zfill(3) + ".ply"
        shutil.move(old, fresh)

    print("ply moving done!")
    shutil.rmtree(base)
    print("fusibile_fused remove done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--thisname', type=str, default='.')
    method = 'mvsnet'

    args = parser.parse_args()

    collect(args.thisname, method)