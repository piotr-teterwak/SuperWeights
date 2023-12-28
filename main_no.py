from __future__ import division
from genericpath import isfile
import os, sys, shutil, time, random
import json
import argparse
import warnings
import contextlib
import copy
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from utils import (AverageMeter, RecorderMeter, time_string, convert_secs2time,
                   Cutout, Lighting, LabelSmoothingNLLLoss, RandomDataset,
                   PrefetchWrapper, fast_collate,
                   get_world_rank, get_world_size, get_local_rank,
                   initialize_dist, get_cuda_device, allreduce_tensor, gather_flops, gather_times, get_flop_range, get_time_range)
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
from tqdm import tqdm
import models
import numpy as np
import random
import PIL
from sklearn.cluster import KMeans
from models.group_gradient_analysis import *
import torch.autograd.profiler as profiler

from timm.data import Mixup, create_transform
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler.cosine_lr import CosineLRScheduler

# Ignore corrupted TIFF warnings
warnings.filterwarnings('ignore', message='.*(C|c)orrupt\sEXIF\sdata.*')
# Ignore anomalous warnings from learning rate schedulers with GradScaler.
warnings.filterwarnings('ignore', message='.*lr_scheduler\.step.*')

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Training script for SuperWeight Ensembles', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data / Model
parser.add_argument('data_path', metavar='DPATH', type=str, help='Path to dataset')
parser.add_argument('--dataset', metavar='DSET', type=str, choices=['cifar10', 'cifar100'], help='Choose between CIFAR-100/10.')
parser.add_argument('--cifar_split', type=str, default='normal', choices=['normal', 'full'], help='Whether to use 90% train, 10% val or 100% train.')
parser.add_argument('--arch', metavar='ARCH', default='swrn', help='model architecture: ' + ' | '.join(model_names) + ' (default: shared wide resnet)')
parser.add_argument('--effnet_arch', metavar='ARCH', default=None, help='EfficientNet architecture type')
parser.add_argument('--depth', type=int, metavar='N', default=28, help='Used for wrn and densenet_cifar')
parser.add_argument('--wide', type=float, metavar='N', default=10, help='Used for growth on densenet cifar, width for wide resnet')
parser.add_argument('--student_widths', type=str, default="2", help='Used for growth on sutdent networks. Separate models with "_"')

# Share params
parser.add_argument('--student_depths', type=str, default="28", help='Depth of the student models. Separate models with "_"')
parser.add_argument('--student_archs', type=str, default=None, help='model architecture: ' + ' | '.join(model_names) + ' (default: shared wide resnet). Split student architectures with ",". On None defaults to arch.')
parser.add_argument('--n_students', type=int, metavar='N', default=1, help='Number of students')
parser.add_argument('--rand_student_train', type=str, default='none', choices=['none', 'epoch', 'iter', 'epoch_n-1', 'iter_n-1'], help='How often to train students')
parser.add_argument('--trans', type=int, metavar='N', default=0, help='Number of kernel transformations to use')
parser.add_argument('--bank_size', type=int, default=8, help='Input > 0 indices maximum number of candidates considered for each layer')
parser.add_argument('--max_params', type=int, default=0, help='Input > 0 indicates maximum parameter size')
parser.add_argument('--group_share_type', type=str, default='emb', choices=['wavg', 'emb'], help='Parameter sharing type for learning groups')
parser.add_argument('--share_type', type=str, default='none', choices=['none', 'sliding_window', 'avg', 'wavg', 'emb', 'avg_slide', 'wavg_slide', 'emb_slide', 'conv'], help='Parameter sharing type')
parser.add_argument('--upsample_type', type=str, default='inter', choices=['none', 'wavg', 'inter', 'linear', 'mask', 'tile', 'repeat'], help='Type of filter upsampling type')
parser.add_argument('--upsample_window', type=int, default=1, help='Number of 3x3 windows to learn upsampling parameters for (not applicible to inter)')
parser.add_argument('--param_groups', type=int, default=-1, help='Number of parameter groups')
parser.add_argument('--param_group_type', type=str, choices=['manual', 'random', 'learned', 'reload'], help='Method for generating parameter groups')
parser.add_argument('--param_group_max_params', type=int, default=5000000, help='Max parameter size for learning parameter groups')
parser.add_argument('--param_group_epochs', type=int, default=15, help='Pretraining epochs for learning parameter groups')
parser.add_argument('--param_group_schedule', type=int, nargs='+', default=[8, 13], help='Learning rate schedule for learning parameter groups')
parser.add_argument('--param_group_gammas', type=int, nargs='+', default=[0.1, 0.1], help='Learning rate drop for learning parameter groups')
parser.add_argument('--param_group_upsample_type', type=str, default='inter', choices=['inter', 'linear', 'mask', 'tile', 'repeat'], help='Type of filter upsampling for learning parameter groups')

parser.add_argument('--param_group_bins', type=int, default=-1, help='Number of bins which each share parameters')
parser.add_argument('--param_group_bin_type', type=str, default='all_nets_groupslim_depth', choices=['depth', 'lpb', 'all_nets_depth', 'all_nets_lpb', 'all_nets_slim_depth', 'all_nets_slim_lpb', 'all_nets_groupslim_depth', 'all_nets_groupslim_lpb'], help='Type of binning if param_group_bins is positive and param_groups == -1')
parser.add_argument('--separate_kernels', default=True, help='Whether to not separate 1x1 and 3x3 kernels into different groups')
parser.add_argument('--param_allocation_normalized', default=False, action='store_true', help='Whether to normalize parameter group sizing')
parser.add_argument('--share_linear', default=False, action='store_true', help='Whether to share linear layers.')

# Coefficient Sharing
parser.add_argument('--coefficient_share', default=False, action='store_true', help='Whether to share coefficients from the start.')
parser.add_argument('--coefficient_unshare_epochs', type=int, default=0, help='Number of epochs before analyzing the gradients to unshare coefficients. If 0 then wont run.')
parser.add_argument('--coefficient_unshare_epoch_gap', type=int, default=0, help='Number of epochs between analyzing the gradients to unshare coefficients.')
parser.add_argument('--coefficient_unshare_threshold', type=float, default=0.5, help='Cosine similarity threshold when analyzing group coefficient gradients.')

# Pretraining
parser.add_argument('--group_split_epochs', type=int, default=0, help='Number of epochs to train before splitting the groups. If 0 then wont run.')
parser.add_argument('--group_split_threshold_start', type=float, default=0.1, help='Starter threshold when analyzing gradient similarity. Loops decrementing by group_split_threshold_decrement until max_params is reached.')
parser.add_argument('--group_split_threshold_decrement', type=float, default=0.05, help='Amount to decrement threshold if parameter budget is not possible with given threshold. See group_split_threshold_start arg.')
parser.add_argument('--group_split_coeff_threshold', default=None, type=float, help='If set then pretrain all students. Students check if gradient similarity is above this threshold to share coeff or get their own.')
parser.add_argument('--group_split_only', default=False, action='store_true', help='Only pretrain.')
parser.add_argument('--group_split_concat_weightwgrad', default=False, action='store_true', help='Whether to concatenate the gradients with weights when computing group split similarities. If false then shares all coefficients within the group and only analyzes gradients, not weights.')

# Optimization
parser.add_argument('--epochs', metavar='N', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--drop_last', default=False, action='store_true', help='Drap last small batch')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--no_nesterov', default=False, action='store_true', help='Disable Nesterov momentum')
parser.add_argument('--exponential_decay', default=False, action='store_true', help='Use an exponential decay schedule')
parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing (default: 0.0)')
parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'rmsproptf'],
                    help='Optimization algorithm (default: SGD)')

# default params used for swrn
parser.add_argument('--schedule', type=int, nargs='+', default=None, help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=None, help='LR is multiplied by gamma on schedule')
parser.add_argument('--warmup_epochs', type=int, default=None, help='Use a linear warmup')
parser.add_argument('--base_lr', type=float, default=0.1, help='Starting learning rate for warmup')
# Step-based schedule used for EfficientNets.
parser.add_argument('--step_size', type=int, default=None, help='Step size for StepLR')
parser.add_argument('--step_gamma', type=float, default=None, help='Decay rate for StepLR')
parser.add_argument('--step_warmup', type=int, default=None, help='Number of warmup steps')

#Regularization
# default for swrn
parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay (L2 penalty).')
parser.add_argument('--no_bn_decay', default=False, action='store_true', help='No weight decay on batchnorm')
parser.add_argument('--cutout', dest='cutout', action='store_true', help='Enable cutout augmentation')
parser.add_argument('--ema_decay', type=float, default=None, help='Elastic model averaging decay')

parser.add_argument('--no_depthwise_decay', default=False, action='store_true', help='No weight decay on depthwise convolutions')

# Checkpoints
parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='Print frequency, minibatch-wise (default: 200)')
parser.add_argument('--save_path', type=str, default='./snapshots/', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='Path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='Manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='Evaluate model on test set')
parser.add_argument('--best_loss', default=False, action='store_true', help='Checkpoint best val loss instead of accuracy (default: no)')

# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
parser.add_argument('--dist', default=False, action='store_true', help='Use distributed training (default: no)')
parser.add_argument('--amp', default=False, action='store_true', help='Use automatic mixed precision (default: no)')
parser.add_argument('--no_dp', default=False, action='store_true', help='Disable using DataParallel (default: no)')

# Random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--job-id', type=str, default='')

# Logging
parser.add_argument('--verbose', action='store_true', help='Excess logging regarding groups.')

args = parser.parse_args()
args.use_cuda = (args.ngpu > 0 or args.dist) and torch.cuda.is_available()

job_id = args.job_id
args.save_path = args.save_path + job_id
result_png_path = './results/' + job_id + '.png'
if not os.path.isdir('results') and get_world_rank() == 0:
    os.mkdir('results')

if get_world_rank() == 0:
    print(str(args))

if args.manualSeed is None: args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda: torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True

if args.dist:
    initialize_dist(f'./init_{args.job_id}')

best_acc = 0
best_los = float('inf')

def load_dataset():
    if args.dataset == 'cifar10':
        mean, std = [x / 255 for x in [125.3, 123.0, 113.9]],  [x / 255 for x in [63.0, 62.1, 66.7]]
        dataset = dset.CIFAR10
        num_classes = 10
    elif args.dataset == 'cifar100':
        mean, std = [x / 255 for x in [129.3, 124.1, 112.4]], [x / 255 for x in [68.2, 65.4, 70.4]]
        dataset = dset.CIFAR100
        num_classes = 100

    mixup_fn = None

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)])
        if args.cutout: train_transform.transforms.append(Cutout(n_holes=1, length=16))
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

        # Ensure only one rank downloads
        if args.dist and get_world_rank() != 0:
            torch.distributed.barrier()

        if args.evaluate or args.cifar_split == 'full':
            train_data = dataset(args.data_path, train=True,
                                 transform=train_transform, download=True)
            test_data = dataset(args.data_path, train=False,
                                transform=test_transform, download=True)

            train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
            test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
        else:
            # partition training set into two instead.
            # note that test_data is defined using train=True
            train_data = dataset(args.data_path, train=True,
                                 transform=train_transform, download=True)
            test_data = dataset(args.data_path, train=True,
                                transform=test_transform, download=True)

            indices = list(range(len(train_data)))
            np.random.shuffle(indices)
            split = int(0.9 * len(train_data))
            train_indices, test_indices = indices[:split], indices[split:]
            if args.dist:
                # Use the distributed sampler here.
                train_subset = torch.utils.data.Subset(
                    train_data, train_indices)
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_subset, num_replicas=get_world_size(),
                    rank=get_world_rank())
                train_loader = torch.utils.data.DataLoader(
                    train_subset, batch_size=args.batch_size,
                    sampler=train_sampler, num_workers=args.workers,
                    pin_memory=True)
                test_subset = torch.utils.data.Subset(test_data, test_indices)
                test_sampler = torch.utils.data.distributed.DistributedSampler(
                    test_subset, num_replicas=get_world_size(),
                    rank=get_world_rank())
                test_loader = torch.utils.data.DataLoader(
                    test_subset, batch_size=args.batch_size,
                    sampler=test_sampler, num_workers=args.workers,
                    pin_memory=True)
            else:
                train_sampler = SubsetRandomSampler(train_indices)
                train_loader = torch.utils.data.DataLoader(
                    train_data, batch_size=args.batch_size,
                    num_workers=args.workers, pin_memory=True,
                    sampler=train_sampler)
                test_sampler = SubsetRandomSampler(test_indices)
                test_loader = torch.utils.data.DataLoader(
                    test_data, batch_size=args.batch_size,
                    num_workers=args.workers, pin_memory=True,
                    sampler=test_sampler)

        # Let ranks through.
        if args.dist and get_world_rank() == 0:
            torch.distributed.barrier()

    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    return num_classes, train_loader, test_loader, mixup_fn


def load_model(num_classes, log, max_params, share_type, upsample_type,
               groups=None, coeff_share_idxs=None, coeff_share=False):

    student_depths = [int(d) for d in args.student_depths.split('_')]
    assert len(student_depths) <= args.n_students or args.n_students == 0
    if len(student_depths) < args.n_students:
        assert len(student_depths) == 1
        student_depths = student_depths * args.n_students

    if args.student_archs is not None:
        student_widths = [float(d) for d in args.student_widths.split('_')]
        width = int(args.wide)
    else:
        student_widths = [int(d) for d in args.student_widths.split('_')]
        width = int(args.wide)
    assert len(student_widths) <= args.n_students or args.n_students == 0
    if len(student_widths) < args.n_students:
        assert len(student_widths) == 1
        student_widths = student_widths * args.n_students

    if args.student_archs is not None:
        student_archs = args.student_archs.split(',')
        if len(student_archs) < args.n_students:
            assert len(student_archs) == 1
            student_archs = student_archs * args.n_students

        # adjust any floats to ints where needed when using multiple archs
        for i, arch in enumerate(student_archs):
            student_widths[i] = int(student_widths[i])

    # Default to teacher architecture
    else:
        student_archs = [args.arch] * args.n_students

    teach_variant = None
    student_variants = [None] * args.n_students

    if 'all_nets' not in args.param_group_bin_type:
        layer_shapes = None
    else:
        layer_shapes = get_layer_shapes(width, num_classes, student_depths, student_widths, student_archs, log)

    print_log("=> creating model '{}'".format(args.arch), log)

    net_coeff_share_idxs = None if coeff_share_idxs is None or 0 not in coeff_share_idxs else coeff_share_idxs[0]
    net = models.__dict__[args.arch](
        share_type, upsample_type, args.upsample_window, args.depth,
        width, args.bank_size, args.max_params, num_classes, groups, args.trans, params=None, param_group_bins=args.param_group_bins, bin_type=args.param_group_bin_type, separate_kernels=args.separate_kernels, allocation_normalized=args.param_allocation_normalized,
        share_linear=args.share_linear, share_coeff=coeff_share, coeff_share_idxs=net_coeff_share_idxs,
        layer_shapes=(0,layer_shapes), variant=teach_variant)

    student_nets = []
    if args.n_students > 0:
        if net.bank:
            params = [net.bank.get_params()]
        else:
            params = None
        student_nets = []
        for i in range(args.n_students):
            net_coeff_share_idxs = None if coeff_share_idxs is None else coeff_share_idxs[i+1]
            student_net = models.__dict__[student_archs[i]](
                share_type, upsample_type, args.upsample_window, student_depths[i],
                student_widths[i], args.bank_size, args.max_params, num_classes, groups, args.trans, params, param_group_bins=args.param_group_bins, bin_type=args.param_group_bin_type, separate_kernels=args.separate_kernels, allocation_normalized=args.param_allocation_normalized,
                share_linear=args.share_linear, share_coeff=coeff_share, coeff_share_idxs=net_coeff_share_idxs,
                layer_shapes=(i+1,layer_shapes), variant=student_variants[i])
            student_nets.append(student_net)

            if params is not None and student_net.bank is not None:
                params.append(student_net.bank.get_params())

        distributed_student_nets = []
        if args.dist:
            for student_net in student_nets:
                distributed_student_nets.append(student_net.to(get_cuda_device()))
        else:
            for student_net in student_nets:
                distributed_student_nets.append(torch.nn.DataParallel(
                    student_net.cuda(), device_ids=list(range(args.ngpu))))

        student_nets = distributed_student_nets
        del distributed_student_nets

    depths = [len(layer_shapes[i]) for i in range(len(layer_shapes))]

    if args.verbose:
        print_log("=> network :\n {}".format(net), log)
    if args.dist:
        net = net.to(get_cuda_device())
    else:
        net = torch.nn.DataParallel(
            net.cuda(), device_ids=list(range(args.ngpu)))

    trainable_params = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([p.numel() for p in trainable_params])
    print_log("Number of parameters: {:,}".format(params), log)

    for student_net in student_nets:
        trainable_params = filter(lambda p: p.requires_grad, student_net.parameters())
        params = sum([p.numel() for p in trainable_params])
        print_log("Student number of parameters: {:,}".format(params), log)
    return net, student_nets, depths


def get_layer_shapes(width, num_classes, student_depths, student_widths, student_archs, log):
    print_log("=> Getting layer shapes ... ", log)
    layer_shapes = [[]]
    unshared = models.__dict__[args.arch](
        'none', None, None, args.depth, width, None, 0, num_classes, None, None,
        params=None, param_group_bins=None, bin_type=None, separate_kernels=False,
        allocation_normalized=False, share_linear=False, share_coeff=False, coeff_share_idxs=None, layer_shapes=None)
    j = 0
    for name, module in unshared.named_modules():
        if args.share_linear:
            if not isinstance(module, (nn.Conv2d, nn.Linear)): continue
        else:
            if not isinstance(module, (nn.Conv2d)): continue
        shape = module.weight.shape
        if len(shape) == 1: continue
        layer_shapes[0].append((0, j, shape))
        j += 1

    print_log('Depth {}'.format(len(layer_shapes[0])), log)
    del unshared

    if args.n_students > 0:
        for i in range(args.n_students):
            layer_shapes.append([])
            unshared = models.__dict__[student_archs[i]](
                'none', None, None, student_depths[i],
                student_widths[i], None, 0, num_classes, None, None,
                params=None, param_group_bins=None, bin_type=None,
                separate_kernels=False, allocation_normalized=False, share_linear=False, share_coeff=False, coeff_share_idxs=None, layer_shapes=None)
            j = 0
            for name, module in unshared.named_modules():
                if args.share_linear:
                    if not isinstance(module, (nn.Conv2d, nn.Linear)): continue
                else:
                    if not isinstance(module, (nn.Conv2d)): continue
                shape = module.weight.shape
                if len(shape) == 1: continue
                layer_shapes[i+1].append((i+1, j, shape))
                j += 1

            print_log('Depth {}'.format(len(layer_shapes[i+1])), log)
            del unshared

    torch.cuda.empty_cache()
    return layer_shapes


def learn_parameter_groups(train_loader, state, num_classes, log):
    print_log('Pretraining to learn parameter groups', log)
    net, student_nets, depths = load_model(num_classes, log, args.max_params, args.share_type,
                    args.upsample_type, groups=-1, coeff_share=False, coeff_share_idxs=None)

    num_warmup = args.param_group_epochs
    schedule = args.param_group_schedule
    gammas = args.param_group_gammas

    decay_skip = ['coefficients']
    if args.no_bn_decay:
        decay_skip.append('bn')
    params = group_weight_decay(net, student_nets, state['decay'], decay_skip)

    if args.label_smoothing > 0.0:
        criterion = LabelSmoothingNLLLoss(args.label_smoothing).cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        params, state['learning_rate'], momentum=state['momentum'],
        nesterov=(not args.no_nesterov and state['momentum'] > 0.0))

    if args.step_size:
        if schedule:
            raise ValueError('Cannot combine regular and step schedules')
        step_scheduler = torch.optim.lr_scheduler.StepLR(
           optimizer, args.step_size, args.step_gamma)
        if args.step_warmup:
            step_scheduler = models.efficientnet.GradualWarmupScheduler(
                optimizer, multiplier=1.0, warmup_epoch=args.step_warmup,
                after_scheduler=step_scheduler)
    else:
        step_scheduler = None

    cos_scheduler = None

    start_time = time.time()
    epoch_time = AverageMeter()
    train_los = -1

    for epoch in range(args.start_epoch, num_warmup):
        current_learning_rate = adjust_learning_rate(optimizer, epoch, gammas, schedule, train_los, cos_scheduler)
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (num_warmup-epoch))

        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, num_warmup, need_time, current_learning_rate), log)

        train_acc, train_los = train(train_loader, net, student_nets, criterion, optimizer, epoch, log, mixup_fn=None, cos_scheduler=cos_scheduler)

        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    # Grab coefficients
    kernel_coefficients = {}
    kernel_layers = {} # { kernel: { net: [layer_idx_in_kernel_coefficients] } }
    kernel_net_layers = {}
    kernel_idxs = {}
    for i, member in enumerate([net] + student_nets):
        net_idx = 0
        for name, module in member.named_modules():
            if not isinstance(module, (models.layers.SConv2d)): continue

            kernel = '{}_{}'.format(module.shape[-1], module.shape[-1])
            if kernel not in kernel_coefficients:
                kernel_coefficients[kernel] = []
                kernel_layers[kernel] = {}
                kernel_net_layers[kernel] = {}
                kernel_idxs[kernel] = 0
            if i not in kernel_layers[kernel]:
                kernel_layers[kernel][i] = []
                kernel_net_layers[kernel][i] = []

            kernel_coefficients[kernel].append(module.coefficients[0].coefficients.data)
            kernel_layers[kernel][i].append(kernel_idxs[kernel])
            kernel_idxs[kernel] += 1
            kernel_net_layers[kernel][i].append(net_idx)
            net_idx += 1

    coefficients = torch.stack(kernel_coefficients['3_3']).cpu().numpy()
    kmeans = KMeans(n_clusters=args.param_groups).fit(coefficients)

    next_group = np.max(kmeans.labels_) + 1

    layer2group = []
    for kernel in kernel_layers.keys():
        if kernel != '3_3':
            g = next_group
            next_group += 1

        for i in range(len(student_nets) + 1):
            if len(layer2group) <= i:
                layer2group.append([-1]*depths[i])

            for j, idx in enumerate(kernel_layers[kernel][i]):
                if kernel == '3_3':
                    layer2group[i][kernel_net_layers[kernel][i][j]] = kmeans.labels_[idx]
                else:
                    layer2group[i][kernel_net_layers[kernel][i][j]] = g

    print(layer2group)
    print()
    print(kmeans.labels_)

    del net
    return layer2group


def get_random_parameter_groups():
    if args.arch == 'swrn':
        num_layers = 29
    elif args.arch == 'swrn_imagenet':
        num_layers = 56
    else:
        raise ValueError('Do not know number of layers for arch')
    groups = np.random.randint(args.param_groups,
                               size=(num_layers - args.param_groups))
    groups = list(groups) + list(range(args.param_groups))
    np.random.shuffle(groups)
    return groups


def get_manual_parameter_groups():
    if args.arch == 'swrn':
        groups = [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 9,
                  10, 11, 12, 12, 12, 12, 12, 12, 13]
    elif args.arch == 'swrn_imagenet':
        groups = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 8, 9, 10, 11, 12, 13, 14,
                  15, 16, 14, 15, 16, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                  21, 22, 23, 21, 22, 23, 21, 22, 23, 21, 22, 23, 24, 25, 26,
                  27, 28, 29, 30, 28, 29, 30, 31]
    else:
        raise ValueError('Do not know manual groups for arch')
    return groups


def get_parameter_groups(train_loader, state, num_classes, log):
    if args.param_group_type == 'manual':
        return get_manual_parameter_groups()
    if args.param_group_type == 'random':
        return get_random_parameter_groups()
    if args.param_group_type == 'learned':
        return learn_parameter_groups(train_loader, state, num_classes, log)
    if args.param_group_type == 'reload':
        groups = np.load(os.path.join(
            args.save_path, 'groups.npy'))
        assert len(set(groups)) == args.param_groups
        return groups
    raise ValueError(
        f'Unknown parameter group type {args.param_group_type}')


def learn_gradient_similarity_groups(train_loader, mixup_fn, state, num_classes, log):
    # Load from checkpoint
    if args.resume:
        return -1, None

    # If we have already pretrained then load in groups and coeff share indexes
    if os.path.isfile(os.path.join(args.save_path, 'groups.npy')) and \
        os.path.isfile(os.path.join(args.save_path, 'coeff_idxs.txt')):

        layer2group = np.load(os.path.join(
                        args.save_path, 'groups.npy'), allow_pickle=True)
        layer2group = layer2group.tolist()
        layer_coeff_share_idxs = json.load(open(os.path.join(
                                    args.save_path, 'coeff_idxs.txt')))
        int_keys = {}
        for nkey in layer_coeff_share_idxs.keys():
            int_keys[int(nkey)] = {}
            for lkey in layer_coeff_share_idxs[nkey].keys():
                int_keys[int(nkey)][int(lkey)] = layer_coeff_share_idxs[nkey][lkey]
        layer_coeff_share_idxs = int_keys
        return layer2group, layer_coeff_share_idxs

    share_coff = args.coefficient_share
    coeff_share_idxs = None
    if not args.group_split_concat_weightwgrad:
        coeff_share_idxs = {}
        for net in range(args.n_students + 1):
            coeff_share_idxs[net] = {}
            for layer in range(args.depth + 1):
                coeff_share_idxs[net][layer] = {'layer': 0, 'net': 0}

    net, student_nets, depths = load_model(num_classes, log, args.max_params, args.share_type,
                                  args.upsample_type, groups=-1, coeff_share=share_coff, coeff_share_idxs=coeff_share_idxs)


    decay_skip = ['coefficients']
    if args.no_bn_decay:
        decay_skip.append('bn')
    params = group_weight_decay(net, student_nets, state['decay'], decay_skip)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        params, state['learning_rate'], momentum=state['momentum'],
        nesterov=(not args.no_nesterov and state['momentum'] > 0.0))

    if args.step_size:
        if args.schedule:
            raise ValueError('Cannot combine regular and step schedules')
        step_scheduler = torch.optim.lr_scheduler.StepLR(
           optimizer, args.step_size, args.step_gamma)
        if args.step_warmup:
            step_scheduler = models.efficientnet.GradualWarmupScheduler(
                optimizer, multiplier=1.0, warmup_epoch=args.step_warmup,
                after_scheduler=step_scheduler)
    else:
        step_scheduler = None

    cos_scheduler = None

    start_time = time.time()
    epoch_time = AverageMeter()
    train_los = -1

    if args.group_split_coeff_threshold is not None:
        pretrain_students = student_nets

    else:
        # Save time during pretraining
        pretrain_students = []
        unique_depths = set([depths[0]])
        for i in range(1,len(depths)):
            if depths[i] in unique_depths: continue
            unique_depths.add(depths[i])
            pretrain_students.append(student_nets[i-1])

    for epoch in range(args.start_epoch, args.group_split_epochs):
        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule, train_los, cos_scheduler)
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.group_split_epochs-epoch))

        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.group_split_epochs, need_time, current_learning_rate), log)

        train_acc, train_los = train(train_loader, net, pretrain_students, criterion, optimizer, epoch, log, mixup_fn=mixup_fn, cos_scheduler=cos_scheduler)

        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    layer2group, layer_coeff_share_idxs = analyze_group_gradients(train_loader, num_classes, log, net, student_nets, criterion, optimizer, depths)

    # Write to files
   # np.save(os.path.join(
   #         args.save_path, 'groups.npy'),
   #         np.array(layer2group))
   # json.dump(layer_coeff_share_idxs,
   #         open(os.path.join(
   #         args.save_path, 'coeff_idxs.txt'),'w'))

    del net
    del student_nets[:]
    del pretrain_students[:]
    del optimizer
    del params
    torch.cuda.empty_cache()

    return layer2group, layer_coeff_share_idxs


def main():
    global best_acc, best_los

    if get_world_rank() == 0:
        if not os.path.isdir(args.save_path):
            os.makedirs(args.save_path)
        log = open(os.path.join(
            args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    else:
        log = None
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("Python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("PyTorch  version : {}".format(torch.__version__), log)
    print_log("CuDNN  version : {}".format(torch.backends.cudnn.version()), log)
    print_log(f'Ranks: {get_world_size()}', log)
    print_log(f'Global batch size: {args.batch_size*get_world_size()}', log)

    if get_world_rank() == 0 and not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    num_classes, train_loader, test_loader, mixup_fn = load_dataset()
    groups = args.param_groups
    layer_coeff_share_idxs = None
    if args.param_groups > 1:
        fn = os.path.join(args.save_path, 'groups.npy')
        if args.evaluate or args.resume:
            groups = np.load(fn, allow_pickle=True)
            groups = groups.tolist()
        else:
            groups = get_parameter_groups(train_loader, state, num_classes, log)
            if args.param_group_type != 'reload' and get_world_rank() == 0:
                np.save(fn, groups)
            if args.param_group_type == 'random':
                # Need to load this from rank 0 to get consistent view.
                torch.distributed.barrier()
                if get_world_rank() != 0:
                    groups = np.load(fn)
        print_log('groups- ' + ', '.join(
            [str(i) + ':' + str(g) for i, g in enumerate(groups)]), log)
    elif args.group_split_epochs > 0:
        groups, layer_coeff_share_idxs = learn_gradient_similarity_groups(train_loader, mixup_fn, state, num_classes, log)
        if args.group_split_only:
            return

    coeff_share = False if args.evaluate else args.coefficient_share
    if coeff_share and args.coefficient_unshare_epochs == 0 or coeff_share and args.group_split_epochs == 0:
        layer_coeff_share_idxs = {}
        for net in range(args.n_students + 1):
            layer_coeff_share_idxs[net] = {}
            for layer in range(args.depth + 1):
                layer_coeff_share_idxs[net][layer] = {'layer': 0, 'net': 0}

    net, student_nets, depths = load_model(num_classes, log, args.max_params, args.share_type,
                                  args.upsample_type, groups=groups, coeff_share_idxs=layer_coeff_share_idxs, coeff_share=coeff_share)

    decay_skip = ['coefficients']
    if args.no_bn_decay:
        decay_skip.append('bn')

    params = group_weight_decay(net, student_nets, state['decay'], decay_skip)
    if args.label_smoothing > 0.0:
        criterion = LabelSmoothingNLLLoss(args.label_smoothing).cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        params, state['learning_rate'], momentum=state['momentum'],
        nesterov=(not args.no_nesterov and state['momentum'] > 0.0))

    if args.step_size:
        if args.schedule:
            raise ValueError('Cannot combine regular and step schedules')
        step_scheduler = torch.optim.lr_scheduler.StepLR(
           optimizer, args.step_size, args.step_gamma)
        if args.step_warmup:
            step_scheduler = models.efficientnet.GradualWarmupScheduler(
                optimizer, multiplier=1.0, warmup_epoch=args.step_warmup,
                after_scheduler=step_scheduler)
    else:
        step_scheduler = None

    cos_scheduler = None

    recorder = RecorderMeter(args.epochs)
    if args.resume:
        if args.resume == 'auto':
            args.resume = os.path.join(args.save_path, 'checkpoint.pth.tar')
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(
                args.resume,
                map_location=get_cuda_device() if args.ngpu else 'cpu')
            recorder = checkpoint['recorder']
            if args.cifar_split == 'normal':
                recorder.refresh(args.epochs)
            args.start_epoch = checkpoint['epoch']
            if 'param_groups' in checkpoint:
                groups = checkpoint['param_groups']
                # reload model if groups were learned
                if type(groups) == list:
                    coeff_share = False if args.evaluate else args.coefficient_share
                    coeff_share_idxs = None
                    net, student_nets, depths = load_model(num_classes, log, args.max_params, args.share_type,
                                                args.upsample_type, groups=groups, coeff_share_idxs=coeff_share_idxs, coeff_share=coeff_share)

            # Hack to load models that were wrapped in (D)DP.
            if args.no_dp:
                net = torch.nn.DataParallel(net, device_ids=[get_local_rank()])
            net.load_state_dict(checkpoint['state_dict'])
            if args.no_dp:
                net = net.module

            # Hack to load models that were wrapped in (D)DP.
            for i, student_net in enumerate(student_nets):
                if args.no_dp:
                    student_net = torch.nn.DataParallel(student_net, device_ids=[get_local_rank()])
                student_net.load_state_dict(checkpoint['student_state_dict'][i])
                if args.no_dp:
                    student_net = student_net.module

            # optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = recorder.max_accuracy(False)
            print_log(
                "=> loaded checkpoint '{}' accuracy={} (epoch {})" .format(
                    args.resume, best_acc, checkpoint['epoch']), log)
        else:
            print_log(
                "=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log(
            "=> do not use any checkpoint for {} model".format(args.arch), log)

    if args.evaluate:
        net.eval()
        for student_model in student_nets:
            student_model.eval()

        ParamCounter([net] + student_nets).log_params()
        input, _ = next(iter(train_loader))
        flops = gather_flops(input.cuda(non_blocking=True), net, student_nets)
        flop_range = get_flop_range(flops)
        time_range = get_time_range(input, net, student_nets)

        if get_world_size() > 1:
            raise RuntimeError('Do not validate with distributed training')

        validate(test_loader, net, student_nets, criterion, log, flop_range=flop_range, time_range=time_range)
        if args.verbose and args.share_type != 'none':
            explore_coefficients(train_loader, net, student_nets, criterion, optimizer)

        return

    start_time = time.time()
    epoch_time = AverageMeter()
    train_los = -1

    prof = False
    if prof:
        profile_models(train_loader, net, student_nets)
        return

    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule, train_los, cos_scheduler)
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))

        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        if not args.cifar_split == 'full':
            print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, current_learning_rate) \
                        + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)
        else:
            print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, current_learning_rate), log)
        analyze_grads = False
        train_acc, train_los = train(train_loader, net, student_nets, criterion, optimizer, epoch, log, analyze_grads, mixup_fn, cos_scheduler)
        if not args.cifar_split == 'full':
            val_acc, val_los = validate(test_loader, net, student_nets, criterion, log)
            recorder.update(epoch, train_los, train_acc, val_los, val_acc)

        if args.coefficient_unshare_epochs > 0 and (epoch + 1 == args.coefficient_unshare_epochs or (args.coefficient_unshare_epoch_gap > 0 and epoch + 1 > args.coefficient_unshare_epochs and (epoch + 1) % args.coefficient_unshare_epoch_gap == 0)):
            print_log('==> Coefficient Gradient Analysis at epoch {}'.format(epoch), log)
            optimizer = analyze_gradients(train_loader, net, student_nets, criterion, optimizer, state, reinit=False)

        if not args.cifar_split == 'full':
            is_best = False
            if args.best_loss:
                if val_los < best_los:
                    is_best = True
                    best_los = val_los
            else:
                if val_acc > best_acc:
                    is_best = True
                    best_acc = val_acc
        elif args.cifar_split == 'full' and epoch == args.epochs - 1:
            is_best = True

        if (not args.cifar_split == 'full') or (args.cifar_split == 'full' and epoch == args.epochs - 1):
            if get_world_rank() == 0:
                chpt_dict = {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': net.state_dict(),
                    'student_state_dict' : [],
                    'param_groups': groups,
                    'recorder': recorder,
                    'optimizer': optimizer.state_dict()
                }
                for student_net in student_nets:
                    chpt_dict['student_state_dict'].append(student_net.state_dict())
                save_checkpoint(chpt_dict, is_best, args.save_path, 'checkpoint.pth.tar')

        epoch_time.update(time.time() - start_time)
        start_time = time.time()

        if get_world_rank() == 0:
            recorder.plot_curve(result_png_path)

    if args.verbose and args.group_split_epochs > 0:
        print_log('Final groups: {}'.format(groups), log)

    if get_world_rank() == 0:
        log.close()


def analyze_gradients(train_loader, model, student_models, criterion, optimizer, state, reinit=False):
    # Unshare coefficients by looking at coeff gradients

    model.eval()
    for student_model in student_models:
        student_model.eval()

    grads = [None] * (len(student_models) + 1)
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(input)

        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        grads[0] = model.module.bank.get_grads(grads[0])

        for j, student_model in enumerate(student_models):
            output_student = student_model(input)
            loss = criterion(output_student, target)
            optimizer.zero_grad()
            loss.backward()
            grads[j+1] = student_model.module.bank.get_grads(grads[j+1])

    new_coeff = model.module.bank.compare_grads(grads, threshold=args.coefficient_unshare_threshold)
    if reinit:
        model.module.bank.reinitialize_params()
        for m in model.module.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
        for student_model in student_models:
            student_model.module.bank.reinitialize_params()
            for m in student_model.module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    torch.nn.init.kaiming_normal_(m.weight)
                    m.bias.data.zero_()
        decay_skip = ['coefficients']
        if args.no_bn_decay:
            decay_skip.append('bn')
        params = group_weight_decay(model, student_models, state['decay'], decay_skip)
        optimizer = torch.optim.SGD(
            params, state['learning_rate'], momentum=state['momentum'],
            nesterov=(not args.no_nesterov and state['momentum'] > 0.0))

    else:
        optimizer.add_param_group({'params': new_coeff, 'weight_decay': 0.})

    return optimizer


def analyze_group_gradients(train_loader, num_classes, log, model, student_models, criterion, optimizer, depths):
    # Split up groups after pretraining by analyzing layer weight gradients

    # Layers are non leaf variables so need to add hooks
    models = [model] + student_models
    for i, net in enumerate(models):
        net.eval()
        for _, module in net.named_modules():
            if not hasattr(module, 'group_id'): continue
            module.register_hook = True

    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        for j, net in enumerate(models):
            HOOK.update_model(j)
            output = net(input)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
        break

    num_params = 0
    threshold = args.group_split_threshold_start
    i = 0
    while i == 0 or num_params > args.max_params:

        if i > 0:
            # layer group_id's were changed so reload model
            model, student_models, _ = load_model(num_classes, log, args.max_params, args.share_type,
                                                args.upsample_type, groups=-1)
            models = [model] + student_models

        print_log('==> Group Gradient Analysis: Threshold  {}'.format(threshold), log)
        next_group = args.param_group_bins + 1
        group2layer, layer2group, num_groups, group_ids, max_group_sizes, layer_coeff_share_idxs = assign_groups(HOOK.grads, models,
                                                                                        depths, next_group,
                                                                                        threshold=threshold,
                                                                                        coeff_threshold=args.group_split_coeff_threshold,
                                                                                        concat_weightwgrad=args.group_split_concat_weightwgrad,
                                                                                        verbose=args.verbose)
        threshold -= args.group_split_threshold_decrement

        num_params = 0
        for _, size in max_group_sizes.items():
            num_params += size
        if args.verbose:
            print_log('num_params {}'.format(num_params), log)
            print_log('max_group_sizes {}'.format(max_group_sizes), log)
        i += 1

    if args.verbose:
        print_log('------> Final', log)
        print_log('layer2group {}'.format(layer2group), log)

    HOOK.update_model(-1)

    for i, net in enumerate(models):
        for name, module in net.named_modules():
            if not hasattr(module, 'group_id'): continue
            module.register_hook = False

    HOOK.remove_grads()

    return layer2group, layer_coeff_share_idxs


def train(train_loader, model, student_models, criterion, optimizer, epoch, log, analyze_grads=False, mixup_fn=None, cos_scheduler=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    print_loss = args.verbose and epoch == 0
    if print_loss: torch.autograd.set_detect_anomaly(True)

    num_steps = len(train_loader)

    model.train()
    for student_model in student_models:
        student_model.train()

    if args.rand_student_train == 'epoch_n-1':
        if args.n_students == 1:
            stu_idxs = [0]
        elif args.n_students > 1:
            stu_idxs = np.random.permutation(np.arange(args.n_students))[:-1]

    grads = [None] * (len(student_models) + 1)
    num_grad_grab_iters = 200
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if mixup_fn is not None:
            input, target = mixup_fn(input, target)

        #with autocast(enabled=args.amp):
        output = model(input)
        if analyze_grads and i < num_grad_grab_iters:
            grads[0] = model.module.bank.get_grads(grads[0])

        loss = criterion(output, target)
        if print_loss:
            # if torch.isnan(loss):
            print_log('output {}, max {}'.format(output, torch.max(output)), log)
            print_log(loss, log)

        output = torch.nn.functional.softmax(output, -1)

        if args.rand_student_train == 'none':
            for j, student_model in enumerate(student_models):
                output_student = student_model(input)
                if analyze_grads and i < num_grad_grab_iters:
                    grads[j+1] = student_model.bank.get_grads(grads[j+1])

                output += torch.nn.functional.softmax(output_student, -1)
                stu_loss = criterion(output_student, target)
                loss += stu_loss

                if print_loss:
                    print_log(stu_loss, log)

        elif args.rand_student_train == 'iter_n-1':
            # Choose random student(s)
            if args.n_students == 1:
                stu_idxs = [0]
            elif args.n_students > 1:
                stu_idxs = np.random.permutation(np.arange(args.n_students))[:-1]
            for stu_idx in stu_idxs:
                student_model = student_models[stu_idx]
                output_student = student_model(input)
                output += torch.nn.functional.softmax(output_student, -1)
                loss += criterion(output_student, target)

        elif args.rand_student_train == 'epoch_n-1':
            for stu_idx in stu_idxs:
                student_model = student_models[stu_idx]
                output_student = student_model(input)
                output += torch.nn.functional.softmax(output_student, -1)
            loss += criterion(output_student, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if print_loss:
                torch.autograd.set_detect_anomaly(False)
                print_loss = False
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                          epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)

    print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)

    if analyze_grads:
        print_log('==> Coefficient Gradient Analysis at epoch {}'.format(epoch), log)
        new_coeff = model.module.bank.compare_grads(grads, threshold=args.coefficient_unshare_threshold)
        optimizer.add_param_group({'params': new_coeff, 'weight_decay': 0.})

    return top1.avg, losses.avg


def validate(val_loader, model, student_models, criterion, log,
             ema_model=None, ema_manager=None, flop_range=None, time_range=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_teacher = AverageMeter()
    top5_teacher = AverageMeter()
    top1_students, top5_students = [], []
    disagreement = AverageMeter()
    criterion = torch.nn.CrossEntropyLoss().cuda()

    for i in range(len(student_models)):
        top1_students.append(AverageMeter())
        top5_students.append(AverageMeter())

    if flop_range is not None:
        for i in range(len(flop_range)):
            flop_range[i].append(AverageMeter())
            flop_range[i].append(AverageMeter())

    if time_range is not None:
        for i in range(len(time_range)):
            time_range[i].append(AverageMeter())
            time_range[i].append(AverageMeter())

    if ema_model is not None:
        ema_model.module.load_state_dict(ema_manager.state_dict())
        model = ema_model

    model.eval()
    for student_model in student_models:
        student_model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # Teacher
            outputs = []
            prec1_students, prec5_students = [], []
            output = torch.nn.functional.softmax(model(input), -1)
            outputs.append(output)
            loss = criterion(output, target)
            prec1_teacher, prec5_teacher = accuracy(output, target, topk=(1, 5))

            # Students
            for student_model in student_models:
                output_student = torch.nn.functional.softmax(student_model(input), -1)
                outputs.append(output_student)
                prec1_student, prec5_student = accuracy(output_student, target, topk=(1, 5))
                prec1_students.append(prec1_student)
                prec5_students.append(prec5_student)

            prec1, prec5 = accuracy(torch.sum(torch.stack(outputs), 0), target, topk=(1, 5))

            batch_time.update(time.time() - end)

            if flop_range is not None:
                prec1_ens_full_range_flop, prec5_ens_full_range_flop = [], []
                for flops in flop_range:
                    idxs = flops[1]
                    flop_output = [outputs[idxs[0]]]
                    if len(idxs) > 1:
                        for idx in idxs[1:]:
                            flop_output.append(outputs[idx])
                    prec1_ens, prec5_ens = accuracy(torch.sum(torch.stack(flop_output), 0), target, topk=(1, 5))
                    prec1_ens_full_range_flop.append(prec1_ens)
                    prec5_ens_full_range_flop.append(prec5_ens)

            if time_range is not None:
                prec1_ens_full_range_time, prec5_ens_full_range_time = [], []
                for times in time_range:
                    idxs = times[1]
                    time_output = [outputs[idxs[0]]]
                    if len(idxs) > 1:
                        for idx in idxs[1:]:
                            time_output.append(outputs[idx])
                    prec1_ens, prec5_ens = accuracy(torch.sum(torch.stack(time_output), 0), target, topk=(1, 5))
                    prec1_ens_full_range_time.append(prec1_ens)
                    prec5_ens_full_range_time.append(prec5_ens)

            div = diversity(outputs)
            disagreement.update(div, input.size(0))

            if args.dist:
                reduced_loss = allreduce_tensor(loss.data)
                reduced_prec1 = allreduce_tensor(prec1)
                reduced_prec5 = allreduce_tensor(prec5)
                losses.update(reduced_loss.item(), input.size(0))
                top1.update(reduced_prec1.item(), input.size(0))
                top5.update(reduced_prec5.item(), input.size(0))
                reduced_prec1 = allreduce_tensor(prec1_teacher)
                reduced_prec5 = allreduce_tensor(prec5_teacher)
                top1_teacher.update(reduced_prec1.item(), input.size(0))
                top5_teacher.update(reduced_prec5.item(), input.size(0))


                for j in range(len(student_models)):
                    reduced_prec1 = allreduce_tensor(prec1_students[j])
                    reduced_prec5 = allreduce_tensor(prec5_students[j])
                    top1_students[j].update(reduced_prec1.item(), input.size(0))
                    top5_students[j].update(reduced_prec5.item(), input.size(0))

                if flop_range is not None:
                    for j in range(len(flop_range)):
                        reduced_prec1 = allreduce_tensor(prec1_ens_full_range_flop[j])
                        reduced_prec5 = allreduce_tensor(prec5_ens_full_range_flop[j])
                        flop_range[j][2].update(reduced_prec1.item(), input.size(0))
                        flop_range[j][3].update(reduced_prec5.item(), input.size(0))

                if time_range is not None:
                    for j in range(len(time_range)):
                        reduced_prec1 = allreduce_tensor(prec1_ens_full_range_time[j])
                        reduced_prec5 = allreduce_tensor(prec5_ens_full_range_time[j])
                        time_range[j][2].update(reduced_prec1.item(), input.size(0))
                        time_range[j][3].update(reduced_prec5.item(), input.size(0))

            else:
                losses.update(loss.data.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
                top5.update(prec5.item(), input.size(0))
                top1_teacher.update(prec1_teacher.item(), input.size(0))
                top5_teacher.update(prec5_teacher.item(), input.size(0))
                for j in range(len(student_models)):
                    top1_students[j].update(prec1_students[j].item(), input.size(0))
                    top5_students[j].update(prec5_students[j].item(), input.size(0))

                if flop_range is not None:
                    for j in range(len(flop_range)):
                        flop_range[j][2].update(prec1_ens_full_range_flop[j].item(), input.size(0))
                        flop_range[j][3].update(prec5_ens_full_range_flop[j].item(), input.size(0))

                if time_range is not None:
                    for j in range(len(time_range)):
                        time_range[j][2].update(prec1_ens_full_range_time[j].item(), input.size(0))
                        time_range[j][3].update(prec5_ens_full_range_time[j].item(), input.size(0))

            end = time.time()

    print_log('  **Test**  Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f} Loss {losses.avg:.5f} ms/sample {times:.3f} Diversity {diversity:.5f}'.format(top1=top1, top5=top5, error1=100-top1.avg, losses=losses, times=batch_time.avg * 1e3 / args.batch_size, diversity=disagreement.avg / (1-top1.avg/100.)), log)

    member_metric_str = '  **Test**  *Member1* Prec@1 {top1_teacher.avg:.3f} Prec@5 {top5_teacher.avg:.3f}'.format(top1_teacher=top1_teacher, top5_teacher=top5_teacher)
    print_log(member_metric_str, log)

    if flop_range is not None:
        print()
        print('=> FLOPS')
        for flops in flop_range:
            print_log('  **Test** Models {models}, Flops {model_flops},  Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(models=flops[1], model_flops=flops[0], top1=flops[2], top5=flops[3]), log)

        print()
        for flops in flop_range:
            print(str(flops[0]) + ', ', end='')
        print()
        for flops in flop_range:
            print('{top1.avg:.3f}, '.format(top1=flops[2]), end='')
        print()

    if time_range is not None:
        print('=> TIMES')
        for times in time_range:
            print_log('  **Test** Models {models}, Time {model_times},  Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(models=times[1], model_times=times[0], top1=times[2], top5=times[3]), log)
        print()
        for times in time_range:
            print(str(times[0]) + ', ', end='')
        print()
        for times in time_range:
            print('{top1.avg:.3f}, '.format(top1=times[2]), end='')
        print()

    return top1.avg, losses.avg


def explore_coefficients(train_loader, model, student_models, criterion, optimizer, log):
    # prints coefficients of each group for verification of sharing/unsharing
    model.eval()
    for student_model in student_models:
        student_model.eval()

    grads = [None] * (len(student_models) + 1)
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        grads[0] = model.module.bank.get_grads(grads[0], explore=True)

        for j, student_model in enumerate(student_models):
            output_student = student_model(input)
            loss = criterion(output_student, target)
            optimizer.zero_grad()
            loss.backward()
            print_log('-----> Net {}'.format(j+1), log)
            grads[j+1] = student_model.module.bank.get_grads(grads[j+1], explore=True)

        return


def profile_models(train_loader, model, student_models):
    model.train()
    for student_model in student_models:
        student_model.train()

    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(input)
        with profiler.profile(with_stack=True, profile_memory=True, record_shapes=True, use_cuda=True) as prof:
            with profiler.record_function("Model"):
                output = model(input)
        print(prof.key_averages(group_by_stack_n=3).table(sort_by='cuda_time_total', row_limit=10))

        break



def print_log(print_string, log):
    if get_world_rank() != 0:
        return  # Only print on rank 0.
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename):
    if get_world_rank() != 0:
        return
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)


def adjust_learning_rate(optimizer, epoch, gammas, schedule, loss, cos_scheduler=None, step_update=False):
    if not args.warmup_epochs and not schedule and not args.exponential_decay and cos_scheduler is None:
        return args.learning_rate  # Bail out here.
    if cos_scheduler is not None:
        if step_update:
            cos_scheduler.step_update(epoch)
        return optimizer.param_groups[0]['lr']
    elif args.warmup_epochs is not None and epoch <= args.warmup_epochs:
        incr = (args.learning_rate - args.base_lr) / args.warmup_epochs
        lr = args.base_lr + incr*epoch
    elif args.exponential_decay:
        lr = args.learning_rate
        assert len(gammas) == 1
        gamma = gammas[0]
        for _ in range(epoch):
            lr = lr * gamma
    else:
        lr = args.learning_rate
        assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



def group_weight_decay(net, student_nets, weight_decay, skip_list=()):

    depthwise_groups = set()
    if args.no_depthwise_decay and args.share_type != 'none':
        # Grab all names of banks for depthwise convolutions
        for name, module in net.named_modules():
            if hasattr(module, 'group_id') and module.shape[1] == 1:
                depthwise_groups.add(module.group_id)

        for student_net in student_nets:
            for name, module in student_net.named_modules():
                if hasattr(module, 'group_id') and module.shape[1] == 1:
                    depthwise_groups.add(module.group_id)

    decay, no_decay = [], []
    # Keep track of params added otherwise we add banks/coefficients multiple times
    params_added = set()
    for name, param in net.named_parameters():
        if not param.requires_grad: continue
        if param in params_added: continue
        params_added.add(param)

        if args.no_depthwise_decay:
            if args.share_type == 'none' and len(list(param.size())) == 4 and list(param.size())[1] == 1:
                no_decay.append(param)
                continue
            elif '_params' in name and int(name.split('.')[-3].split('_')[3]) in depthwise_groups:
                no_decay.append(param)
                continue

        if sum([pattern in name for pattern in skip_list]) > 0:
            no_decay.append(param)
        else:
            decay.append(param)

    for student_net in student_nets:
        skip_list = list(skip_list)
        for name, param in student_net.named_parameters():
            if not param.requires_grad: continue
            if param in params_added: continue
            params_added.add(param)

            if args.no_depthwise_decay:
                if args.share_type == 'none' and len(list(param.size())) == 4 and list(param.size())[1] == 1:
                    no_decay.append(param)
                    continue
                elif '_params' in name and int(name.split('.')[-3].split('_')[3]) in depthwise_groups:
                    no_decay.append(param)
                    continue

            if sum([pattern in name for pattern in skip_list]) > 0:
                no_decay.append(param)
            else:
                decay.append(param)

    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]


def accuracy(output, target, topk=(1,)):
    if len(target.shape) > 1: return torch.tensor(1), torch.tensor(1)

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))
    return res


def diversity(member_logits):
    dis = []
    with torch.no_grad():
        for i in range(len(member_logits)):
            preds_1 = torch.argmax(member_logits[i], axis=-1)
            for j in range(i+1,len(member_logits)):
                preds_2 = torch.argmax(member_logits[j], axis=-1)
                dis.append(torch.sum(preds_1 != preds_2))
    return torch.mean(torch.Tensor(dis)) / member_logits[0].shape[0]


# Easy way to count all model parameters including bn, fc, and coefficients
class ParamCounter(nn.Module):
    def __init__(self, models):
        super(ParamCounter, self).__init__()
        self.models = nn.ModuleList(models)

    def log_params(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([p.numel() for p in trainable_params])
        print('Total parameters {}'.format(params))



if __name__ == '__main__':
    main()
