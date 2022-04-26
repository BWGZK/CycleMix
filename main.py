import os
import argparse
import datetime
import random
import json
import time
from pathlib import Path
from tensorboardX import SummaryWriter
import clr
import numpy as np
import torch
from copy import deepcopy
torch.multiprocessing.set_sharing_strategy('file_system')

from torch.utils.data import DataLoader, DistributedSampler
import data
#from mmdet import datasets
import util.misc as utils
from data import build
from engines import train_one_epoch
from inference import infer, evaluate
from models import build_model


def get_args_parser():
    # define task, label values, and output channels
    tasks = {
        'MR': {'lab_values': [0, 1, 2, 3, 4, 5], 'out_channels': 4}
        }
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=6000, type=int)
    parser.add_argument('--lr_drop', default=2000, type=int)
    parser.add_argument('--tasks', default=tasks, type=dict)
    parser.add_argument('--model', default='MSCMR', required=False)

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--in_channels', default=1, type=int)
    
    # Puzzle Mix
    parser.add_argument('--in_batch', type=str2bool, default=False, help='whether to use different lambdas in batch')
    parser.add_argument('--mixup_alpha', type=float, help='alpha parameter for mixup')
    parser.add_argument('--box', type=str2bool, default=False, help='true for CutMix')
    parser.add_argument('--graph', type=str2bool, default=False, help='true for PuzzleMix')
    parser.add_argument('--neigh_size', type=int, default=4, help='neighbor size for computing distance beteeen image regions')
    parser.add_argument('--n_labels', type=int, default=3, help='label space size')

    parser.add_argument('--beta', type=float, default=1.2, help='label smoothness')
    parser.add_argument('--gamma', type=float, default=0.5, help='data local smoothness')
    parser.add_argument('--eta', type=float, default=0.2, help='prior term')

    parser.add_argument('--transport', type=str2bool, default=True, help='whether to use transport')
    parser.add_argument('--t_eps', type=float, default=0.8, help='transport cost coefficient')
    parser.add_argument('--t_size', type=int, default=-1, help='transport resolution. -1 for using the same resolution with graphcut')

    parser.add_argument('--adv_eps', type=float, default=10.0, help='adversarial training ball')
    parser.add_argument('--adv_p', type=float, default=0.0, help='adversarial training probability')

    parser.add_argument('--clean_lam', type=float, default=0.0, help='clean input regularization')
    parser.add_argument('--mp', type=int, default=8, help='multi-process for graphcut (CPU)')


    # * Loss coefficients
    parser.add_argument('--multiDice_loss_coef', default=0, type=float)
    parser.add_argument('--CrossEntropy_loss_coef', default=1, type=float)
    parser.add_argument('--Rv', default=1, type=float)
    parser.add_argument('--Lv', default=1, type=float)
    parser.add_argument('--Myo', default=1, type=float)
    parser.add_argument('--Avg', default=1, type=float)
    
    # dataset parameters
    parser.add_argument('--dataset', default='MSCMR_scribble_v2', type=str,
                        help='multi-sequence CMR segmentation dataset')
    # set your outputdir 
    parser.add_argument('--output_dir', default='/data/MSCMR_cycleMix/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', type=str,
                        help='device to use for training / testing')
    parser.add_argument('--GPU_ids', type=str, default = '5', help = 'Ids of GPUs')    
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', default = False , action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):
    writer = SummaryWriter(log_dir=args.output_dir + '/summary')
    args.mean = torch.tensor([0.5], dtype=torch.float32).reshape(1,1,1,1).cuda()
    args.std = torch.tensor([0.5], dtype=torch.float32).reshape(1,1,1,1).cuda()

    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors, visualizer = build_model(args)
    model.to(device)
    print(model)

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [{"params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad]}]
    optimizer = torch.optim.Adam(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    print('Building training dataset...')
    dataset_train_dict = build(image_set='train', args=args)
    num_train = [len(v) for v in dataset_train_dict.values()]
    print('Number of training images: {}'.format(sum(num_train)))

    print('Building validation dataset...')
    dataset_val_dict = build(image_set='val', args=args)
    num_val = [len(v) for v in dataset_val_dict.values()]
    print('Number of validation images: {}'.format(sum(num_val)))


    sampler_train_dict = {k : torch.utils.data.RandomSampler(v) for k, v in dataset_train_dict.items()}
    sampler_val_dict = {k : torch.utils.data.SequentialSampler(v) for k, v in dataset_val_dict.items()}

    batch_sampler_train = { 
        k : torch.utils.data.BatchSampler(v, args.batch_size, drop_last=True) for k, v in sampler_train_dict.items()
        }
    dataloader_train_dict = {
        k : DataLoader(v1, batch_sampler=v2, collate_fn=utils.collate_fn, num_workers=args.num_workers) 
        for (k, v1), v2 in zip(dataset_train_dict.items(), batch_sampler_train.values())
        }
    dataloader_val_dict = {
        k : DataLoader(v1, args.batch_size, sampler=v2, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers) 
        for (k, v1), v2 in zip(dataset_val_dict.items(), sampler_val_dict.values())
        }

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.whst.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        # if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        if not args.eval and 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        infer(model, criterion, dataloader_train_dict, device)

    print("Start training")
    start_time = time.time()
    best_dic = None
    best_dice = None
    for epoch in range(args.start_epoch, args.epochs):
        # use cyclic learning rate
        # optimizer.param_groups[0]['lr'] = clr.cyclic_learning_rate(epoch, mode='exp_range', gamma=1)
        train_stats = train_one_epoch(model, criterion, dataloader_train_dict, optimizer, device, epoch,args,writer)
        ## lr_scheduler
        lr_scheduler.step()

        test_stats = evaluate(
            model, criterion, postprocessors, dataloader_val_dict, device, args.output_dir, visualizer, epoch, writer
        )

        # save checkpoint for high dice score
        dice_score = test_stats["Avg"]
        print("dice score:", dice_score)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if best_dice == None or dice_score > best_dice:
                best_dice = dice_score
                print("Update best model!")
                checkpoint_paths.append(output_dir / 'best_checkpoint.pth')

            # You can change the threshold
            if dice_score > 0.50:
                print("Update high dice score model!")
                file_name = str(dice_score)[0:6]+'new_checkpoint.pth'
                checkpoint_paths.append(output_dir / file_name)

            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MSCMR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU_ids)
    main(args)
