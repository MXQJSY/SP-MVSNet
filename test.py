"""
2022/03/12, doubleZ, PKU
Testing process.
"""

import os, time, logging
import torch
import torch.nn.parallel
from torch.utils.data import DataLoader
from collections import OrderedDict

from datasets.dtu import DTUDataset
from datasets.tnt import TNTDataset

from datasets.datasets_io import *
from models.losses import *

from utils.utils import *
from utils.opts import get_opts
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.backends.cudnn.benchmark = True

args = get_opts()

logger = None

# @Note Dataset
if args.which_dataset == "dtu":
    test_dataset = DTUDataset(
        args.root_dir, args.list_file, "test", 
        args.n_views, args.n_pyramids,
        args.n_depths
    )
elif args.which_dataset == "tnt":
    test_dataset = TNTDataset(
        args.root_dir, args.list_file, "test", 
        args.n_views, args.n_pyramids,
        args.n_depths,
        rescale=args.rescale,
        max_w=args.max_w, max_h=args.max_h
    )

test_dataloader = DataLoader(
    test_dataset, 
    args.batch_size, 
    shuffle=False, 
    num_workers=16, 
    drop_last=False,
    pin_memory=True
)

# test_dataloader = list(test_dataloader)

# @Note Model
from models.odmvs import MVSNet
    
model = MVSNet(
    which_dataset=args.which_dataset,
    base_channels=args.base_channels
)
model.cuda()


def test():
    model.eval()

    with torch.no_grad():
        lastname = ""
        for batch_idx, sample in enumerate(test_dataloader):
            # if batch_idx < 129: continue 
            sample_cuda = tocuda(sample)
            torch.cuda.empty_cache()

            imgs = sample_cuda["imgs"]
            proj_matrices_pyramid = sample_cuda["proj_matrices_pyramid"]
            camera_parameter = sample_cuda["camera_parameter"]
            output_dirs = sample["output_dirs"]

            start_time = time.time()
            # @Note test main
            outputs = model(
                imgs.float(), proj_matrices_pyramid,
                camera_parameter["extrinsics_matrices"], camera_parameter["intrinsics_matrices_pyramid"], 
                sample_cuda["init_depth_hypos"], 
                args.depth_mode, 
                is_training=False
            )
            print("memory:"+str(torch.cuda.max_memory_allocated()/(1024**2))+"MB")
            depth_est_pyramid_, confidence_pyramid_ = outputs['depth_est_pyramid'], outputs['confidence_pyramid']
            del sample_cuda, imgs

            thisname = sample["output_dirs"][0].split('/')[0]
            if thisname != lastname:
                logger.info('{}------------------------'.format(thisname))
                lastname = thisname

            logger.info('Test Iter {}/{}, time={:.3f}'.format(batch_idx, len(test_dataloader), time.time()-start_time))
            
            for output_dir, depth_est, confidence in zip(output_dirs, tensor2numpy(depth_est_pyramid_[-1]), tensor2numpy(confidence_pyramid_[-1])):
                depth_filename = os.path.join(args.outdir, output_dir.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(args.outdir, output_dir.format('confidence', '.pfm'))
                os.makedirs(depth_filename.rsplit('/',1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/',1)[0], exist_ok=True)

                save_pfm(depth_filename, depth_est)
                save_pfm(confidence_filename, confidence)
                
def fast_test():
    # model.eval()

    with torch.no_grad():
        lastname = ""
        times = args.instance_test
        for batch_idx, sample in enumerate(test_dataloader):
            # if batch_idx < 129: continue 
            if batch_idx == times:
                times+=49
                sample_cuda = tocuda(sample)
                torch.cuda.empty_cache()

                imgs = sample_cuda["imgs"]
                proj_matrices_pyramid = sample_cuda["proj_matrices_pyramid"]
                camera_parameter = sample_cuda["camera_parameter"]
                output_dirs = sample["output_dirs"]

                start_time = time.time()
                # @Note test main
                outputs = model(
                    imgs.float(), proj_matrices_pyramid,
                    camera_parameter["extrinsics_matrices"], camera_parameter["intrinsics_matrices_pyramid"], 
                    sample_cuda["init_depth_hypos"], 
                    args.depth_mode, 
                    is_training=False
                )
                depth_est_pyramid_, confidence_pyramid_ = outputs['depth_est_pyramid'], outputs['confidence_pyramid']
                del sample_cuda, imgs

                thisname = sample["output_dirs"][0].split('/')[0]
                if thisname != lastname:
                    logger.info('{}------------------------'.format(thisname))
                    lastname = thisname

                logger.info('Test Iter {}/{}, time={:.3f}'.format(batch_idx, len(test_dataloader), time.time()-start_time))
                
                for output_dir, depth_est, confidence in zip(output_dirs, tensor2numpy(depth_est_pyramid_[-1]), tensor2numpy(confidence_pyramid_[-1])):
                    depth_filename = os.path.join(args.outdir, output_dir.format('depth_est', '.pfm'))
                    confidence_filename = os.path.join(args.outdir, output_dir.format('confidence', '.pfm'))
                    os.makedirs(depth_filename.rsplit('/',1)[0], exist_ok=True)
                    os.makedirs(confidence_filename.rsplit('/',1)[0], exist_ok=True)

                    save_pfm(depth_filename, depth_est)
                    save_pfm(confidence_filename, confidence)

def initLogger():
    global logger

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    curTime = time.strftime('%Y%m%d-%H%M', time.localtime(time.time()))
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    logger.info("Logger initialized.")
    logger.info("Current time: {}".format(curTime))

    settings_str = "All settings:\n"
    for k,v in vars(args).items(): 
        settings_str += '{0}: {1}\n'.format(k,v)
    logger.info(settings_str)


def initCkpt():
    global model
    logger.info("loading model {}".format(args.ckpt_path))
    state_dict = torch.load(args.ckpt_path)

    new_state_dict = OrderedDict()
    for k, v in state_dict["model"].items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)


if __name__ == '__main__':
    initLogger()
    initCkpt()
    if args.instance_test == -1:
        test()
    else:
        fast_test()
