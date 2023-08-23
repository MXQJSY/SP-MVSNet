

import argparse

def get_opts():
    parser = argparse.ArgumentParser(description="Multi-View Stereo")

    parser.add_argument('--root_dir', type=str,
                        default='.../Data/MVS/CVP-MVSNet/dtu-train-128/',
                        help='root directory of dtu dataset')
    parser.add_argument('--root_eval_dir', type=str,
                        default='.../Data/MVS/CVP-MVSNet/dtu-test-1200/',
                        help='root directory of dtu dataset in eval during training')
    parser.add_argument('--list_file', type=str,
                        default='datasets/lists/dtu/train.txt')
    parser.add_argument('--list_eval_file', type=str,
                        default='datasets/lists/dtu/test.txt')


    parser.add_argument('--mode', default='train', 
                        choices=["train", "val", "test"])
    parser.add_argument('--which_dataset', default='dtu',
                        choices=['dtu', 'tnt', 'blendedmvs'])

    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=12, help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1)


    parser.add_argument('--n_views', type=int, default=3,
                        help='number of views (including ref) to be used in training')
    parser.add_argument('--n_views_eval', type=int, default=5,
                        help='number of views (including ref) to be used in eval during training')
    

    parser.add_argument('--n_depths', type=int, default=48,
                        help='number of depths of cost volume')
    parser.add_argument('--n_depths_eval', type=int, default=192,
                        help='number of depths of cost volume')
    parser.add_argument('--interval_scale', type=float, default=2.5,
                        help='depth interval scale between each depth step (2.5mm)')

    parser.add_argument('--n_pyramids', type=int, default=2,
                        help='number of pyramids')
    parser.add_argument('--n_pyramids_eval', type=int, default=5,
                        help='number of pyramids')


    parser.add_argument('--base_channels', type=int, default=16)
    parser.add_argument('--depth_mode', type=str, default="regression", choices=["regression", "classification"])


    parser.add_argument('--rescale', action='store_true')
    parser.add_argument('--max_w', type=int, default=1600,
                        help="max width in eval during testing, it should divisible by 2^n_pyramids, the recommended: [1600, 1152]")
    parser.add_argument('--max_h', type=int, default=1184,
                        help="max width in eval during testing, it should divisible by 2^n_pyramids, the recommended: [1184, 864]")


    parser.add_argument('--ckpt_path', type=str, default='',
                        help='pretrained checkpoint path to load')
    parser.add_argument('--ckptdir', default='./checkpoints/', 
                        help='the directory to save logs and checkpoints')
    parser.add_argument('--logdir', default='./logs/', 
                        help='the directory to save text logs')
    parser.add_argument('--outdir', default='./outputs',            
                        help='the directory to save output .pfm and ,ply')
    parser.add_argument('--resume', action="store_true",  
                        help='continue to train the model')

    parser.add_argument('--summary_freq', type=int, default=100, 
                        help='print and summary frequency')
    parser.add_argument('--save_freq', type=int, default=1, 
                        help='save checkpoint frequency')
    parser.add_argument('--start_eval_epoch', type=int, default=0)
    parser.add_argument('--eval_freq', type=int, default=1,
                        help='eval frequency')
    parser.add_argument('--summary_freq_eval', type=int, default=15)
    parser.add_argument('--no_eval', action='store_true')


    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-5,
                        help='weight decay')


    parser.add_argument('--exp_name', type=str, default='setup',
                        help='experiment name')
    parser.add_argument('--seed', type=int, default=42, metavar='S', 
                        help='random seed')
    parser.add_argument('--instance_test', type=int, default=-1, 
                        help='test for real fast')
    parser.add_argument('--lrepochs', type=str, default="6,9,12,15:2", help='epoch ids to downscale lr and the downscale rate')
    parser.add_argument('--temp_epoch', type=int, default=5, help='number of epochs for temperature annealing')
    parser.add_argument('--temp_init', type=float, default=30.0, help='initial value of temperature')
    return parser.parse_args()
