
import os, time, logging, sys, json, csv

import torch
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from datasets.dtu import DTUDataset
from models.losses import *

from utils import *
from utils.opts import get_opts
from utils.utils import *
from utils.warmup import *

args = get_opts()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.benchmark = True

logger, tb_writer = None, SummaryWriter(args.ckptdir)
csv_header, csv_path = [], ""

# @Note Dataset
train_dataset = DTUDataset(
    root_dir=args.root_dir,
    list_file=args.list_file,
    mode='train',
    n_views=args.n_views,
    n_depths=args.n_depths,
    n_pyramids=args.n_pyramids
)

train_dataloader = DataLoader(
    train_dataset,
    shuffle=False if args.exp_name=='debug' else True,
    num_workers=16,
    batch_size=args.batch_size,
    drop_last=True,
    pin_memory=True
)

if not args.no_eval:
    eval_dataset = DTUDataset(
        root_dir=args.root_eval_dir,
        list_file=args.list_eval_file,
        mode='val',
        n_views=args.n_views_eval,
        n_depths=args.n_depths_eval,
        n_pyramids=args.n_pyramids_eval,
        max_w=args.max_w, max_h=args.max_h
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.num_gpus,
        shuffle=False,
        num_workers=16,
        drop_last=True,
        pin_memory=True
    )
    eval_dataloader = list(eval_dataloader)


# @Note Model
from models.odmvs import MVSNet

model = MVSNet(
    which_dataset=args.which_dataset,
    base_channels=args.base_channels
)
model = nn.DataParallel(model)
model.cuda()
model.train()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

start_epoch = None

# @Note Train
def train():
    # origin
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9 ** ((epoch-1) / len(train_dataloader)), last_epoch=len(train_dataloader)*start_epoch-1)


    for epoch_idx in range(start_epoch, args.num_epochs):
        logger.info('Epoch {}:'.format(epoch_idx))
        global_step = len(train_dataloader) * epoch_idx

        losses_batch, abs_errs_batch, acc_2mms_batch = [], [], []

        # @Note train
        # if epoch_idx < args.temp_epoch and hasattr(model, 'net_update_temperature'):
        #     temp = get_temperature(batch_idx + 1, epoch_idx, len(train_dataloader),
        #                            temp_epoch=args.temp_epoch, temp_init=args.temp_init)
        #     model.net_update_temperature(temp)

        for batch_idx, sample in enumerate(train_dataloader):
            start_time = time.time()
            global_step = len(train_dataloader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            losses_sample, abs_errs_sample, acc_2mms_sample = [], [], []

            loss, abs_err, acc_2mm = train_sample(sample, epoch_idx, global_step, do_summary)
            lr_scheduler.step()

            losses_sample.append(loss), losses_batch.append(loss)
            abs_errs_sample.append(abs_err), abs_errs_batch.append(abs_err)
            acc_2mms_sample.append(acc_2mm), acc_2mms_batch.append(acc_2mm)
            
            if do_summary:
                logger.info('Train: Epoch {}/{}, Iter {}/{}, lr={:.3}, loss={:.3f}, abs_error={:2.3f}, acc_2mm={:1.3f}, time={:1.3f}'.format(
                    epoch_idx, args.num_epochs, batch_idx, len(train_dataloader),
                    optimizer.param_groups[0]["lr"],
                    np.mean(losses_sample), np.mean(abs_errs_sample), np.mean(acc_2mms_sample), time.time() - start_time
                ))
        
        # @Note checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>2}.ckpt".format(args.ckptdir, epoch_idx)
            )
            logger.info("Epoch {} checkpoints model_{:0>6}.ckpt saved!".format(epoch_idx, epoch_idx))

        logger.info("Epoch {} train finish: loss={:.3f}, abs_error={:2.3f}, 2mm={:1.3f}".format(epoch_idx, np.mean(losses_batch), np.mean(abs_errs_batch), np.mean(acc_2mms_batch)))


        if not args.no_eval:
            # @Note eval
            if (epoch_idx >= args.start_eval_epoch and (epoch_idx % args.eval_freq == 0)) or (epoch_idx == args.num_epochs-1):
                avg_test_scalars = DictAverageMeter()
                for batch_idx, sample in enumerate(eval_dataloader):
                    start_time = time.time()
                    global_step = len(eval_dataloader) * epoch_idx + batch_idx
                    do_summary = global_step % args.summary_freq_eval == 0
                    loss, scalar_outputs, image_outputs = eval_sample(sample, epoch_idx)

                    if do_summary:
                        save_scalars(tb_writer, 'eval', scalar_outputs, global_step)
                        save_images(tb_writer, 'eval', image_outputs, global_step)

                        logger.info('Eval: Epoch {}/{}, Iter {}/{}, eval_loss={:.3f}, time={:1.3f}'.format(
                            epoch_idx, args.num_epochs, batch_idx, len(eval_dataloader), 
                            loss, time.time() - start_time
                        ))
                    avg_test_scalars.update(scalar_outputs)
                    del scalar_outputs, image_outputs
                logger.info("Epoch {} eval finish".format(epoch_idx))
                with open(csv_path, 'a', newline='', encoding='utf-8') as f: 
                    csv_writer = csv.DictWriter(f, fieldnames=csv_header)
                    csv_writer.writerows([avg_test_scalars.mean()])


    logger.info("Training finish! Last loss={:.3f}, abs_error={:2.3f}, 2mm={:1.3f}".format(np.mean(losses_batch), np.mean(abs_errs_batch), np.mean(acc_2mms_batch)))
    with open(os.path.join(args.ckptdir, "train_res.log"), mode='w', encoding='utf-8') as file:
        train_res = {
            "epoch": args.num_epochs-1,
            "loss": "{:.3f}".format(np.mean(losses_batch)),
            "abs_error": "{:2.3f}".format(np.mean(abs_errs_batch)),
            "2mm": "{:1.3f}".format(np.mean(acc_2mms_batch))
        }
        file.write(json.dumps(train_res))
        

def train_sample(sample, epoch_idx, global_step, detailed_summary=False):
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    imgs = sample_cuda["imgs"]
    proj_matrices_pyramid = sample_cuda["proj_matrices_pyramid"]
    camera_parameter = sample_cuda["camera_parameter"]
    depth_gt_pyramid = sample_cuda["depth_gt_pyramid"]
    mask_pyramid = sample_cuda["mask_pyramid"]

    outputs = model(
        imgs.float(), proj_matrices_pyramid,
        camera_parameter["extrinsics_matrices"], camera_parameter["intrinsics_matrices_pyramid"], 
        sample_cuda["init_depth_hypos"],
        args.depth_mode,
        is_training=True
    )
    
    depth_est_pyramid, confidence_pyramid = outputs["depth_est_pyramid"], outputs["confidence_pyramid"]
    prob_volume_pyramid, depth_hypos_pyramid, interval_pyramid = outputs['prob_volume_pyramid'], outputs['depth_hypos_pyramid'], outputs['interval_pyramid']
    
    losses = []
    for i in range(args.n_pyramids):
        depth_est, depth_gt, mask = depth_est_pyramid[i], depth_gt_pyramid[i], mask_pyramid[i]
        prob_volume, depth_hypos, interval = prob_volume_pyramid[i], depth_hypos_pyramid[i], interval_pyramid[i]
        
        if args.depth_mode == 'regression':
            losses.append(regression_loss(depth_est.float(), depth_gt.float(), mask, epoch_idx, MAX_EPOCH=args.num_epochs))
        elif args.depth_mode == 'classification':
            if interval.shape != torch.Size([]):
                interval = interval[0]
            losses.append(classification_loss(prob_volume, depth_hypos, interval, depth_gt, mask))
    loss = sum(losses)
    
    loss.backward()
    optimizer.step()

    # @Note Tensorboard log
    with torch.no_grad():
        depth_est, depth_gt, mask, confidence = depth_est_pyramid[-1], depth_gt_pyramid[-1], mask_pyramid[-1]>0.5, confidence_pyramid[-1]

        abs_err = abs_error(depth_est, depth_gt, mask)
        acc_2mm = acc_threshold(depth_est, depth_gt, mask, 2)

        if detailed_summary:
            img_ = imgs[0,0,:,:,:].cpu()
            depth_gt_ = visualize_depth(depth_gt[0])
            depth_est_ = visualize_depth(depth_est[0]*mask[0].float())
            error_map_ = visualize_depth((depth_est[0].float()-depth_gt[0].float()).abs()*mask[0].float())
            prob_ = visualize_prob(confidence[0]*mask[0].float())
            stack = torch.stack([img_, depth_gt_, depth_est_, error_map_, prob_])
            tb_writer.add_images('train/image_GT_est_errormap_prob', stack, global_step)
            
            acc_1mm = acc_threshold(depth_est, depth_gt, mask, 1)
            acc_4mm = acc_threshold(depth_est, depth_gt, mask, 4)

            tb_writer.add_scalar('train/epoch', epoch_idx, global_step)
            tb_writer.add_scalar('train/loss', loss, global_step)
            tb_writer.add_scalar('train/abs_err', abs_err, global_step)
            tb_writer.add_scalar('train/acc_1mm', acc_1mm, global_step)
            tb_writer.add_scalar('train/acc_2mm', acc_2mm, global_step)
            tb_writer.add_scalar('train/acc_4mm', acc_4mm, global_step)

    return tensor2float(loss), tensor2float(abs_err), tensor2float(acc_2mm)


@make_nograd_func
def eval_sample(sample, epoch_idx):
    model.eval()

    sample_cuda = tocuda(sample)
    imgs = sample_cuda["imgs"]
    proj_matrices_pyramid = sample_cuda["proj_matrices_pyramid"]
    camera_parameter = sample_cuda["camera_parameter"]
    depth_gt_pyramid = sample_cuda["depth_gt_pyramid"]
    mask_pyramid = sample_cuda["mask_pyramid"]

    outputs = model(
        imgs.float(), proj_matrices_pyramid,
        camera_parameter["extrinsics_matrices"], camera_parameter["intrinsics_matrices_pyramid"], 
        sample_cuda["init_depth_hypos"],
        args.depth_mode,
        is_training=False
    )

    depth_est_pyramid, confidence_pyramid = outputs["depth_est_pyramid"], outputs["confidence_pyramid"]
    prob_volume_pyramid, depth_hypos_pyramid, interval_pyramid = outputs['prob_volume_pyramid'], outputs['depth_hypos_pyramid'], outputs['interval_pyramid']
    
    losses = []
    for i in range(args.n_pyramids):
        depth_est, depth_gt, mask = depth_est_pyramid[i], depth_gt_pyramid[i], mask_pyramid[i]
        prob_volume, depth_hypos, interval = prob_volume_pyramid[i], depth_hypos_pyramid[i], interval_pyramid[i]
        
        if args.depth_mode == 'regression':
            losses.append(regression_loss(depth_est.float(), depth_gt.float(), mask, epoch_idx, MAX_EPOCH=args.num_epochs))
        elif args.depth_mode == 'classification':
            if interval.shape != torch.Size([]):
                interval = interval[0]
            losses.append(classification_loss(prob_volume, depth_hypos, interval, depth_gt, mask))
    loss = sum(losses)

    depth_est, depth_gt, mask = depth_est_pyramid[-1], depth_gt_pyramid[-1], mask_pyramid[-1]>0.5
    scalar_outputs = {
        'epoch': float(epoch_idx),
        'loss': loss,
        'abs_err': abs_error(depth_est, depth_gt, mask),
        'acc_2mm': acc_threshold(depth_est, depth_gt, mask, 2),
        'acc_4mm': acc_threshold(depth_est, depth_gt, mask, 4),
        'acc_8mm': acc_threshold(depth_est, depth_gt, mask, 8),

        "abserr_2mm": AbsDepthError_metrics(depth_est, depth_gt, mask, [0, 2.0]),
        "abserr_4mm": AbsDepthError_metrics(depth_est, depth_gt, mask, [2.0, 4.0]),
        "abserr_8mm": AbsDepthError_metrics(depth_est, depth_gt, mask, [4.0, 8.0]),
    }

    image_outputs = {
        'depth_est': depth_est * mask.float(),
        'depth_est_nomask': depth_est,
        'depth_gt': depth_gt,
        'errormap': (depth_est - depth_gt).abs() * mask.float(),
        'ref_img': imgs[:,0,:,:,:],
    }

    return tensor2float(loss), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


def initLogger():
    global logger
    global csv_header, csv_path

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    curTime = time.strftime('%Y%m%d-%H%M', time.localtime(time.time()))
    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)
    logfile = os.path.join(args.logdir, curTime + '.log')
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fileHandler = logging.FileHandler(logfile, mode='a')
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    logger.info("Logger initialized.")
    logger.info("Writing logs to file: {}".format(logfile))
    logger.info("Current time: {}".format(curTime))

    if not args.no_eval:
        csv_header = ['epoch', 'loss', 'abs_err', 'acc_2mm', 'acc_4mm', 'acc_8mm', 'abserr_2mm', 'abserr_4mm', 'abserr_8mm']
        csv_path = os.path.join(args.ckptdir, "eval.csv")
        if not args.resume:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                csv_writer = csv.DictWriter(f, fieldnames=csv_header)
                csv_writer.writeheader()

    settings_str = "All settings:\n"
    for k,v in vars(args).items(): 
        settings_str += '{0}: {1}\n'.format(k,v)
    logger.info(settings_str)


def initCkpt():
    global model, optimizer, start_epoch

    if not os.path.isdir(args.ckptdir):
        os.mkdir(args.ckptdir)

    assert args.mode == "train"

    start_epoch = 0
    if args.resume:
        saved_models = [fn for fn in os.listdir(args.ckptdir) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # use the latest checkpoint file
        loadckpt = os.path.join(args.ckptdir, saved_models[-1])
        
        state_dict = torch.load(loadckpt)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1

        logger.info("Resuming {}".format(loadckpt))

    logger.info("Start at epoch {}".format(start_epoch))
    logger.info('>>> Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


def get_temperature(iteration, epoch, iter_per_epoch, temp_epoch=10, temp_init=30.0):
    total_temp_iter = iter_per_epoch * temp_epoch
    current_iter = iteration + epoch * iter_per_epoch
    temperature = 1.0 + max(0, (temp_init - 1.0) * ((total_temp_iter - current_iter) / total_temp_iter))
    return temperature

if __name__ == '__main__':
    initLogger()
    initCkpt()

    train()