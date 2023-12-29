import torch
import torch.nn as nn
from math import sqrt

class AffineFlow(nn.Module):
    """ 
    Scales + Shifts the flow by (learned) constants per dimension.
    """
    def __init__(self, data_dim): #we are listing the parameters
        super().__init__()

        self.s = nn.Parameter(torch.randn(data_dim)/sqrt(data_dim))
        self.t = nn.Parameter(torch.randn(data_dim)/sqrt(data_dim))
    
    def forward(self, x):
        z = x * torch.exp(self.s) + self.t
        log_det = torch.sum(self.s)
        return z, log_det
    
    def inverse(self, z):
        x = (z - self.t) * torch.exp(-self.s)
        log_det = torch.sum(-self.s)
        return x, log_det