# -*-coding:utf-8-*-
import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
import math
from torch._jit_internal import weak_module
from torch.nn.parameter import Parameter


class PrunedNetwork(nn.Module):

    def __init__(self):
        super(PrunedNetwork, self).__init__()

