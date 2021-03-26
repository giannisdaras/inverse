import torch
import torch.nn as nn
import numpy as np
import math
import os
import glob
from collections import defaultdict
from torch.nn import functional as F


class BicubicDownSampler(nn.Module):
    def bicubic_kernel(self, x, a=-0.50):
        """
        This equation is exactly copied from the website below:
        https://clouard.users.greyc.fr/Pantheon/experiments/rescaling/index-en.html#bicubic
        """
        abs_x = torch.abs(x)
        if abs_x <= 1.:
            return (a + 2.) * torch.pow(abs_x, 3.) - (a + 3.) * torch.pow(abs_x, 2.) + 1
        elif 1. < abs_x < 2.:
            return a * torch.pow(abs_x, 3) - 5. * a * torch.pow(abs_x, 2.) + 8. * a * abs_x - 4. * a
        else:
            return 0.0
    

def get_inpainting_mask(x):
    mask = torch.ones(x.shape, device=x.device)
    bs, x, y = torch.where(x.sum(dim=1) == -3)
    mask[bs, :, x, y] = 0
    return mask


class ForwardOperator:
    def __call__(self):
        raise NotImplementerError()


class Inpainter(ForwardOperator):
    def __init__(self, mask):
        self.mask = mask
    def __call__(self, x):
        return x * self.mask

