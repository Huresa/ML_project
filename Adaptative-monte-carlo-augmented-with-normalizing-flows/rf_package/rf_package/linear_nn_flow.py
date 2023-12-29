import torch
import torch.nn as nn

class NN_Block(nn.Module):
    def __init__(self, data_dim, hidden_dim=4):
        super(NN_Block, self).__init__()

        self.scale = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim),
            nn.Tanh()
        )

        self.translation = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def forward(self, x):
        scale = self.scale(x)
        translation = self.translation(x)
        y = x * torch.exp(scale) + translation
        log_det = scale.sum()
        return y, log_det
    
    def inverse(self, y):
        scale = self.scale(y)
        translation = self.translation(y)
        x = (y - translation)*torch.exp(-scale)
        log_det = -scale.sum()
        return x, log_det

class BlockFlow(nn.Module):
    def __init__(self, data_dim, num_blocks=4):
        super(BlockFlow, self).__init__()

        self.layers = nn.ModuleList([NN_Block(data_dim) for _ in range(num_blocks)])

    def forward(self, x):
        log_det_sum = 0
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_sum += log_det
        return x, log_det_sum
    
    def inverse(self, z):
        log_det_sum = 0
        for layer in self.layers:
            z, log_det = layer.inverse(z)
            log_det_sum +=  log_det
        return z, log_det_sum