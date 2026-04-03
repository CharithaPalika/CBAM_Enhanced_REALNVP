# RealNVP + CBAM Image Generation

This repository contains a PyTorch implementation of a RealNVP-based normalizing flow for image generation, with optional CBAM attention inside coupling-network residual blocks. 

## What Is Included

- RealNVP model implementation for multi-scale flow transformations.
- Optional CBAM (Channel + Spatial Attention) modules in coupling sub-networks.
- Dataset loader utilities for `cifar10`, `celeba`, `imnet32`, and `imnet64`.
- Training loop with validation monitoring and sample image export.
- Notebooks for experimentation (`run.ipynb` and files under `temp/`).

## Repository Layout

- `real_nvp.py`: Core flow layers, coupling layers, multi-scale architecture, sampling.
- `cbam.py`: CBAM attention blocks.
- `load_data.py`: Dataset construction and preprocessing.
- `utils.py`: Logit transform and helper classes.
- `train.py`: Main training function (`train(args)`).
- `run.ipynb`: Notebook entry point for running experiments.

## Requirements

Python 3.9+ is recommended.

Install core dependencies:

```bash
pip install torch torchvision numpy jupyter
```

If you use CUDA, install a CUDA-compatible PyTorch build from the official PyTorch install page.

## Quick Start

### 1. Prepare Data

The current loader expects these locations:

- CIFAR-10: downloaded automatically to `../../data/CIFAR10`
- CelebA: `/kaggle/input/celeba-dataset/img_align_celeba`
- ImageNet32: `../../data/ImageNet32/train`
- ImageNet64: `../../data/ImageNet64/train`

If you are not running on Kaggle, update the dataset paths in `load_data.py`.

### 2. Configure Output Paths

The training code writes outputs to:

- Samples: `/kaggle/working/samples/<dataset>/...`
- Models: `/kaggle/working/models/<dataset>/...`

For local runs, edit these paths in `train.py`.

### 3. Run Training

There is no command-line parser in `train.py`; training is started by calling `train(args)`.

Example from a Python script or notebook:

```python
from types import SimpleNamespace
from train import train

args = SimpleNamespace(
    dataset="cifar10",      # cifar10 | celeba | imnet32 | imnet64
    batch_size=64,
    base_dim=64,
    res_blocks=8,
    bottleneck=True,
    skip=True,
    weight_norm=True,
    coupling_bn=True,
    affine=True,
    use_cbam=False,
    lr=1e-3,
    momentum=0.9,
    decay=0.999,
    max_epoch=10,
    sample_size=16,
)

metrics = train(args)
```

## Training Arguments Used by `train(args)`

- `dataset`
- `batch_size`
- `base_dim`
- `res_blocks`
- `bottleneck`
- `skip`
- `weight_norm`
- `coupling_bn`
- `affine`
- `use_cbam`
- `lr`
- `momentum`
- `decay`
- `max_epoch`
- `sample_size`
