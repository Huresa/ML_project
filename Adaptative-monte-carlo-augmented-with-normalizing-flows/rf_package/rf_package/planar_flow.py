import torch
import torch.nn as nn
from math import sqrt

class PlanarFlow(nn.Module):

    def __init__(self, data_dim):
        super().__init__()

        self.u = nn.Parameter(torch.rand(data_dim)/sqrt(data_dim))
        self.w = nn.Parameter(torch.rand(data_dim)/sqrt(data_dim))
        self.b = nn.Parameter(torch.rand(1)/sqrt(data_dim))
        self.h = nn.Tanh()
    
    def h_prime(self, z):
        return 1 - self.h(z) ** 2
    
    def constrained_u(self):
        """
        Constrain the parameters u to ensure invertibility
        """
        wu = torch.matmul(self.w.T, self.u)
        m = lambda x: -1 + torch.log(1 + torch.exp(x))
        return self.u + (m(wu) - wu) * (self.w / (torch.norm(self.w) ** 2 + 1e-15))
    
    def forward(self, z):
        u = self.constrained_u()
        hidden_units = torch.matmul(self.w.T, z.T) + self.b

        x = z + u.unsqueeze(0) * self.h(hidden_units).unsqueeze(-1)

        psi = self.h_prime(hidden_units).unsqueeze(0) * self.w.unsqueeze(-1)

        log_det = torch.log((1+torch.matmul(u.T, psi)).abs() + 1e-15)

        return x, log_det

    def inverse(self, x):
        u = self.constrained_u()
        hidden_units = (torch.matmul(self.w.T, x.T) + self.b).T
        z = x - u.unsqueeze(0) * self.h(hidden_units).unsqueeze(-1)
        psi = self.h_prime(hidden_units).unsqueeze(0) * self.w.unsqueeze(-1)
        log_det = -torch.log((1 + torch.matmul(u.T, psi)).abs() + 1e-15)
        return z, log_det

class LayeredPlanarFlow(nn.Module):

    def __init__(self, data_dim, flow_length = 16):
        super().__init__()

        self.layers = nn.Sequential(
            *(PlanarFlow(data_dim) for _ in range(flow_length)))

    def forward(self, z):
        log_det_sum = 0
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_sum += log_det
        return z, log_det_sum
    
    def inverse(self, z):
        log_det_sum = 0
        for layer in self.layers:
            z, log_det = layer.inverse(z)
            log_det_sum += log_det
        return z, log_det_sum