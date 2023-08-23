# -*- coding: UTF-8 -*-
"""
2022/03/12, doubleZ, PKU
Dataset I/O scripts.
"""
import re, os, sys
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms


def read_cam(filename, interval_scale=1.06):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0])
    depth_interval = float(lines[11].split()[1]) * interval_scale

    return intrinsics, extrinsics, depth_min, depth_interval


def read_img(filename):
    img = Image.open(filename)

    img = np.array(img, dtype=np.float32) / 255.

    if img.shape[0] == 1200:    # dtu(test)
        img = img[:-16, :, :]
    elif img.shape[0] == 1080:  # tnt(test)
        img = img[:-24, :, :]

    return img


def read_depth(filename):
    depth = read_pfm(filename)[0]

    if depth.shape[0] == 1200:  # dtu(evaluation)
        depth = depth[:-16, :]

    return np.array(depth, dtype=np.float32)


def read_mask(filename):
    mask = np.array(Image.open(filename), dtype=np.float32)

    if mask.shape[0] == 1200:  # dtu(evaluation)
        mask = mask[:-16, :]

    return mask


def scale_img_intrinsics(img, intrinsics, max_w, max_h):
    h, w = img.shape[:2]
    
    scale_w = 1.0 * max_w / w
    scale_h = 1.0 * max_h / h
    intrinsics[0, :] *= scale_w
    intrinsics[1, :] *= scale_h

    img = cv2.resize(img, (int(max_w), int(max_h)))

    return img, intrinsics


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def save_pfm(filename, image, scale=1):

    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()


def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


def read_pair(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for _ in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            data.append((ref_view, src_views))
    return data