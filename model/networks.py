import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from typing import List


###############################################################################
# Functions
###############################################################################
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


##############################################################################
# Losses
##############################################################################
class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()   
        self.criterion = nn.MSELoss()
    
    def forward(self, x, y):
        return self.criterion(x, y)


##############################################################################
# NN blocks
##############################################################################
class FFNet(nn.Module):
    def __init__(self, dims: List=[6, 13, 13, 4, 1], activation: str="relu"):
        """ 
        dims in paper one: [6, 13, 4, 1] 
        active function in paper one: sigmoid
        """
        super(FFNet, self).__init__()
        sequence = []
        if activation == "sigm":
            for i, dim in enumerate(dims[:-1]):
                sequence += [nn.Linear(dim, dims[i+1]), nn.Sigmoid()]
        elif activation == "relu":
            for i, dim in enumerate(dims[:-1]):
                sequence += [nn.Linear(dim, dims[i+1]), nn.LeakyReLU(0.2)]
        
        self.model = nn.Sequential(*sequence[:-1])

    def forward(self, X):
        return self.model(X)
