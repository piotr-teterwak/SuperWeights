# SuperWeight Networks

Implementation of SuperWeight Networks from "Learning to Compose SuperWeights for Neural Parameter Allocation Search"


## Requirements

```
Python 3, PyTorch == 1.11.0, torchvision == 0.12.0, timm == 0.5.4, fvcore,
matplotlib, sklearn
```


## Training

To train a 4 member Multi-Depth/Width ensemble of WRN 28-[7,4] 16-[7,4] on CIFAR-100 run:

```
python main_no.py path/to/data --dataset cifar100 --job-id <jobid> --ngpu <number of gpus> \
--cifar_split full --epochs 200 --batch_size 128 --decay 5e-4 --schedule 60 120 160 --gammas 0.2 0.2 0.2 --cutout \
--arch swrn --share_type wavg_slide --max_params 18000000 --depth 28 --wide 7 \
--n_students 3 --student_depths 28_16_16 --student_widths 4_7_4 \
--group_split_epochs 20 --group_split_threshold_start 0.1 \
--coefficient_share --coefficient_unshare_epochs 10 --coefficient_unshare_threshold 0.9 --param_group_bins 4
```

The `--depth` and `--wide` flags control the deepest network, `--n_students`, `--student_depths`, and `--student_widths` dictate the remaining networks. Depths must be in decreasing order and for each depth the widths must be in decreasing order. For CIFAR-10, simply adjust the `--dataset` flag.


To train a Multi-Width configuration of WRN 28-[7,4,3] on CIFAR-100 run:

```
python main_no.py path/to/data --dataset cifar100 --job-id <jobid> --ngpu <number of gpus> \
--cifar_split full --epochs 200 --batch_size 128 --decay 5e-4 --schedule 60 120 160 --gammas 0.2 0.2 0.2 --cutout \
--arch swrn --share_type wavg_slide --max_params 18000000 --depth 28 --wide 7 \
--n_students 2 --student_depths 28_28 --student_widths 4_3 \
--group_split_epochs 20 --group_split_threshold_start 0.1 \
--coefficient_share --coefficient_unshare_epochs 10 --coefficient_unshare_threshold 0.9  --param_group_bins 4
```


## Evaluation

To evaluate a trained model with the Multi-Depth/Width configuration, run:

```
python main_no.py path/to/data --dataset cifar100 --job-id <jobid> --ngpu <number of gpus> \
--cifar_split full --epochs 200 --batch_size 128 --decay 5e-4 --schedule 60 120 160 --gammas 0.2 0.2 0.2 --cutout \
--arch swrn --share_type wavg_slide --max_params 18000000 --depth 28 --wide 7 \
--n_students 3 --student_depths 28_16_16 --student_widths 4_7_4 \
--group_split_epochs 20 --group_split_threshold_start 0.1 \
--coefficient_share --coefficient_unshare_epochs 10 --coefficient_unshare_threshold 0.9 \
 --evaluate --resume snapshots/<jobid>/checkpoint.pth.tar    --param_group_bins 4
```

Code largely authored by Soren Nelson and built on from https://github.com/BryanPlummer/SSN.
