
import torch
import torch.nn as nn
import argparse
import torch, torchvision
import torch.distributions as distributions
import torch.optim as optim
import torchvision.utils as utils
import argparse
import torch, torchvision
import torch.distributions as distributions
import torch.optim as optim
import torchvision.utils as utils
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as distributions
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np



def logit_transform(x, constraint=0.9, reverse=False):
    '''Transforms data from [0, 1] into unbounded space.

    Restricts data into [0.05, 0.95].
    Calculates logit(alpha+(1-alpha)*x).

    Args:
        x: input tensor.
        constraint: data constraint before logit.
        reverse: True if transform data back to [0, 1].
    Returns:
        transformed tensor and log-determinant of Jacobian from the transform.
        (if reverse=True, no log-determinant is returned.)
    '''
    if reverse:
        x = 1. / (torch.exp(-x) + 1.)    # [0.05, 0.95]
        x *= 2.             # [0.1, 1.9]
        x -= 1.             # [-0.9, 0.9]
        x /= constraint     # [-1, 1]
        x += 1.             # [0, 2]
        x /= 2.             # [0, 1]
        return x, 0
    else:
        [B, C, H, W] = list(x.size())
        
        # dequantization
        noise = distributions.Uniform(0., 1.).sample((B, C, H, W))
        x = (x * 255. + noise) / 256.
        
        # restrict data
        x *= 2.             # [0, 2]
        x -= 1.             # [-1, 1]
        x *= constraint     # [-0.9, 0.9]
        x += 1.             # [0.1, 1.9]
        x /= 2.             # [0.05, 0.95]

        # logit data
        logit_x = torch.log(x) - torch.log(1. - x)

        # log-determinant of Jacobian from the transform
        pre_logit_scale = torch.tensor(
            np.log(constraint) - np.log(1. - constraint))
        log_diag_J = F.softplus(logit_x) + F.softplus(-logit_x) \
            - F.softplus(-pre_logit_scale)

        return logit_x, torch.sum(log_diag_J, dim=(1, 2, 3))
        
class Hyperparameters():
    def __init__(self, base_dim, res_blocks, bottleneck, 
        skip, weight_norm, coupling_bn, affine, CBAM = False):
        """Instantiates a set of hyperparameters used for constructing layers.

        Args:
            base_dim: features in residual blocks of first few layers.
            res_blocks: number of residual blocks to use.
            bottleneck: True if use bottleneck, False otherwise.
            skip: True if use skip architecture, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
            coupling_bn: True if batchnorm coupling layer output, False otherwise.
            affine: True if use affine coupling, False if use additive coupling.
        """
        self.base_dim = base_dim
        self.res_blocks = res_blocks
        self.bottleneck = bottleneck
        self.skip = skip
        self.weight_norm = weight_norm
        self.coupling_bn = coupling_bn
        self.affine = affine
        self.CBAM = CBAM