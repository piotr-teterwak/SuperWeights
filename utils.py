import os
import sys
import time
import os.path
import warnings
import random
import torch
import torchvision
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from fvcore.nn import FlopCountAnalysis
from numbers import Number
from typing import Any, Callable, List, Optional, Union, Counter, Dict
from itertools import combinations


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


# Lighting data augmentation take from here - https://github.com/eladhoffer/convNet.pytorch/blob/master/preprocess.py
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img))


# Adapted from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/image_classification/smoothing.py
class LabelSmoothingNLLLoss(torch.nn.Module):
    """NLL loss with label smoothing."""

    def __init__(self, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = (-logprobs.gather(dim=-1, index=target.unsqueeze(1))).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence*nll_loss + self.smoothing*smooth_loss
        return loss.mean()


class RandomDataset(torch.utils.data.Dataset):
    """Dataset that just returns a random tensor for debugging."""

    def __init__(self, sample_shape, dataset_size, label=True, pil=False,
                 transform=None):
        super().__init__()
        self.sample_shape = sample_shape
        self.dataset_size = dataset_size
        self.label = label
        self.transform = transform
        if pil:
            d = torch.rand(sample_shape)
            self.d = torchvision.transforms.functional.to_pil_image(d)
        else:
            self.d = torch.rand(sample_shape)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        d = self.d
        if self.transform is not None:
            d = self.transform(d)
        if self.label:
            return d, 0
        else:
            return d


# Adapted from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/image_classification/dataloaders.py#L250
class PrefetchWrapper:
    """Fetch ahead and do some asynchronous processing."""

    def __init__(self, data_loader, mean, stdev, lighting):
        self.data_loader = data_loader
        self.mean = mean
        self.stdev = stdev
        self.lighting = lighting
        self.stream = torch.cuda.Stream()
        self.sampler = data_loader.sampler  # To simplify set_epoch.

    def prefetch_loader(data_loader, mean, stdev, lighting, stream):
        if lighting is not None:
            mean = torch.tensor(mean).cuda().view(1, 3, 1, 1)
            stdev = torch.tensor(stdev).cuda().view(1, 3, 1, 1)
        else:
            mean = torch.tensor([x*255 for x in mean]).cuda().view(1, 3, 1, 1)
            stdev = torch.tensor([x*255 for x in stdev]).cuda().view(1, 3, 1, 1)

        first = True
        for next_input, next_target in data_loader:
            with torch.cuda.stream(stream):
                next_target = next_target.cuda(non_blocking=True)
                next_input = next_input.cuda(non_blocking=True).float()
                if lighting is not None:
                    # Scale and apply lighting first.
                    next_input = next_input.div_(255.0)
                    next_input = lighting(next_input).sub_(mean).div_(stdev)
                else:
                    next_input = next_input.sub_(mean).div_(stdev)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target
        yield input, target

    def __iter__(self):
        return PrefetchWrapper.prefetch_loader(
            self.data_loader, self.mean, self.stdev, self.lighting, self.stream)

    def __len__(self):
        return len(self.data_loader)


def fast_collate(batch):
    if isinstance(batch[0][0], torch.Tensor):
        targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
        assert len(targets) == len(batch)
        tensor = torch.zeros((len(batch), *batch[0][0].shape), dtype=torch.uint8)
        for i in range(len(batch)):
            tensor[i].copy_(batch[i][0])
        return tensor, targets

    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        # Suppress warnings.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


class RecorderMeter(object):
  """Computes and stores the minimum loss value and its epoch index"""
  def __init__(self, total_epoch):
    self.reset(total_epoch)

  def reset(self, total_epoch):
    assert total_epoch > 0
    self.total_epoch   = total_epoch
    self.current_epoch = 0
    self.epoch_losses  = np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_losses  = self.epoch_losses - 1

    self.epoch_accuracy= np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_accuracy= self.epoch_accuracy

  def refresh(self, epochs):
    if epochs == self.total_epoch: return
    self.epoch_losses = np.vstack( (self.epoch_losses, np.zeros((epochs - self.total_epoch, 2), dtype=np.float32) - 1) )
    self.epoch_accuracy = np.vstack( (self.epoch_accuracy, np.zeros((epochs - self.total_epoch, 2), dtype=np.float32)) )
    self.total_epoch = epochs

  def update(self, idx, train_loss, train_acc, val_loss, val_acc):
    assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(self.total_epoch, idx)
    self.epoch_losses  [idx, 0] = train_loss
    self.epoch_losses  [idx, 1] = val_loss
    self.epoch_accuracy[idx, 0] = train_acc
    self.epoch_accuracy[idx, 1] = val_acc
    self.current_epoch = idx + 1
    return self.max_accuracy(False) == val_acc

  def max_accuracy(self, istrain):
    if self.current_epoch <= 0: return 0
    if istrain: return self.epoch_accuracy[:self.current_epoch, 0].max()
    else:       return self.epoch_accuracy[:self.current_epoch, 1].max()

  def plot_curve(self, save_path):
    title = 'the accuracy/loss curve of train/val'
    dpi = 80
    width, height = 1200, 800
    legend_fontsize = 10
    scale_distance = 48.8
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    x_axis = np.array([i for i in range(self.total_epoch)]) # epochs
    y_axis = np.zeros(self.total_epoch)

    plt.xlim(0, self.total_epoch)
    plt.ylim(0, 100)
    interval_y = 5
    interval_x = 5
    plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
    plt.yticks(np.arange(0, 100 + interval_y, interval_y))
    plt.grid()
    plt.title(title, fontsize=20)
    plt.xlabel('the training epoch', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)

    y_axis[:] = self.epoch_accuracy[:, 0]
    plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_accuracy[:, 1]
    plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)


    y_axis[:] = self.epoch_losses[:, 0]
    plt.plot(x_axis, y_axis*50, color='g', linestyle=':', label='train-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_losses[:, 1]
    plt.plot(x_axis, y_axis*50, color='y', linestyle=':', label='valid-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    if save_path is not None:
      fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
      print ('---- save figure {} into {}'.format(title, save_path))
    plt.close(fig)


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(
        ISOTIMEFORMAT, time.gmtime(time.time())))
    return string


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600*need_hour) / 60)
    need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
    return need_hour, need_mins, need_secs


def time_file_str():
    ISOTIMEFORMAT = '%Y-%m-%d'
    string = '{}'.format(time.strftime(
        ISOTIMEFORMAT, time.gmtime(time.time())))
    return string + '-{}'.format(random.randint(1, 10000))


# Utilities for distributed training.

def get_num_gpus():
    """Number of GPUs on this node."""
    return torch.cuda.device_count()


def get_local_rank():
    """Get local rank from environment."""
    if 'MV2_COMM_WORLD_LOCAL_RANK' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_LOCAL_RANK'])
    elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    elif 'SLURM_LOCALID' in os.environ:
        return int(os.environ['SLURM_LOCALID'])
    else:
        return 0


def get_local_size():
    """Get local size from environment."""
    if 'MV2_COMM_WORLD_LOCAL_SIZE' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_LOCAL_SIZE'])
    elif 'OMPI_COMM_WORLD_LOCAL_SIZE' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
    elif 'SLURM_NTASKS_PER_NODE' in os.environ:
        return int(os.environ['SLURM_NTASKS_PER_NODE'])
    else:
        return 1


def get_world_rank():
    """Get rank in world from environment."""
    if 'MV2_COMM_WORLD_RANK' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_RANK'])
    elif 'OMPI_COMM_WORLD_RANK' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        return int(os.environ['SLURM_PROCID'])
    else:
        return 0


def get_world_size():
    """Get world size from environment."""
    if 'MV2_COMM_WORLD_SIZE' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_SIZE'])
    elif 'OMPI_COMM_WORLD_SIZE' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_SIZE'])
    elif 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS'])
    else:
        return 1


def initialize_dist(init_file):
    """Initialize PyTorch distributed backend."""
    torch.cuda.init()
    torch.cuda.set_device(get_local_rank())
    init_file = os.path.abspath(init_file)
    torch.distributed.init_process_group(
        backend='nccl', init_method=f'file://{init_file}',
        rank=get_world_rank(), world_size=get_world_size())
    torch.distributed.barrier()
    # Ensure the init file is removed.
    if get_world_rank() == 0 and os.path.exists(init_file):
        os.unlink(init_file)


def get_cuda_device():
    """Get this rank's CUDA device."""
    return torch.device(f'cuda:{get_local_rank()}')


def allreduce_tensor(t):
    """Allreduce and average tensor t."""
    rt = t.clone().detach()
    torch.distributed.all_reduce(rt)
    rt /= get_world_size()
    return rt


# Flop counting
from model_profiling import model_profiling

def gather_flops(input, net, student_nets):
    Handle = Callable[[List[Any], List[Any]], Union[Counter[str], Number]]
    ops: Dict[str, Handle] = {
        "aten::mul": mul_flop_jit,
        "aten::add": add_flop_jit,
        "aten::sum": add_flop_jit,
        "aten::batch_norm": None # Can be fused at inference time so ignore (matches Slimmable)
    }

    model_flops = []
    flops = FlopCountAnalysis(net, torch.unsqueeze(input[0], 0)).set_op_handle(**ops)
    print("Flops: total {:,}".format(flops.total()))
    model_flops.append(flops.total())
    
    for i in range(len(student_nets)):
        flops = FlopCountAnalysis(student_nets[i], torch.unsqueeze(input[0], 0)).set_op_handle(**ops)
        print("Student {} Flops: total {:,}".format(i, flops.total()))
        model_flops.append(flops.total())

    return model_flops


def get_flop_range(flops):
    model_idxs = np.arange(len(flops))
    flop_range = []

    for i in range(1, len(flops)+1):
        for comb in combinations(zip(model_idxs, flops), i):
            comb_flops, comb_idxs = 0., []
            for model in comb:
                comb_idxs.append(model[0])
                comb_flops += model[1]
            flop_range.append([comb_flops, comb_idxs])
    flop_range = sorted(flop_range)
    return flop_range


def gather_times(input, net, student_nets):
    model_times = []
    times = []
    # burn in
    with torch.no_grad():
        for _ in range(1000):
            output = net(input)

        for _ in range(1000):
            torch.cuda.synchronize()
            start = time.time()
            output = net(input)
            torch.cuda.synchronize()
            end = time.time()
            times.append((end - start)*1e3 / input.shape[0])
        model_times.append(np.average(times))
        print("Model 1 time: {}".format(np.average(times)))

    for i in range(len(student_nets)):
        times = []
        # burn in
        with torch.no_grad():
            for _ in range(1000):
                output = student_nets[i](input)

            for _ in range(1000):
                torch.cuda.synchronize()
                start = time.time()
                output = student_nets[i](input)
                torch.cuda.synchronize()
                end = time.time()
                times.append((end - start)*1e3 / input.shape[0])
        
            model_times.append(np.average(times))
            print("Model {} time: {}".format(i+2, np.average(times)))
    return model_times


def get_time_range_from_times(times):
    return get_flop_range(times)


def get_time_range(input, net, student_nets):
    nets = [net] + student_nets
    model_idxs = np.arange(len(student_nets) + 1)

    final_time = []
    sequential = True

    x = input.cuda(non_blocking=True)

    for i in range(1, len(model_idxs)+1):
        for comb_idxs in combinations(model_idxs, i):
            with torch.no_grad():
                # run the timing
                times = []
                for _ in range(1000):
                    for idx in comb_idxs:
                        output = nets[idx](x)

                if not sequential:
                    output = []
                    start = time.time()
                    for _ in range(1000):
                        for idx in comb_idxs:
                            output.append(nets[idx](x))
                    end = time.time()
                    times.append((end - start)*1e3 / (input.shape[0]*1000))

                else:
                    output = []
                    start = time.time()
                    for _ in range(1000):
                        for idx in comb_idxs:
                            output.append(nets[idx](x))
                            torch.cuda.synchronize()
                    end = time.time()
                    times.append((end - start)*1e3 / (input.shape[0]*1000))

            final_time.append([np.average(times), comb_idxs])
            print("Model {} time: {}".format(comb_idxs, np.average(times)))
    final_time = sorted(final_time)
    return final_time

def mul_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for element wise multiplication.
    """
    # Inputs should be a list of at least length 1.
    # Inputs contains the shape of the matrix.
    input_shapes = [v.type().sizes() for v in inputs]
    output_shapes = [v.type().sizes() for v in outputs]

    assert len(input_shapes) >= 1, input_shapes
    flop = 0.5 * np.prod(input_shapes[0]) # larger of the input shapes
    return flop

def add_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for add. Handles both sum and element-wise addition.
    """
    # Inputs should be a list of at least length 1.
    # Inputs contains the shape of the matrix.
    input_shapes = []
    for v in inputs:
        if str(v.type()) == 'Tensor':
            input_shapes.append(v.type().sizes())

    output_shapes = []
    for v in outputs:
        if str(v.type()) == 'Tensor':
            input_shapes.append(v.type().sizes())

    assert len(input_shapes) >= 1, input_shapes
    flop = 0.5 * np.prod(input_shapes[0])
    return flop