import torch

def U(configuration, energy_parameters):
    a, L = energy_parameters
    N = configuration.shape[0] // 2
    shifts = torch.arange(2, 2 * N, 2) # Create a tensor of shifts
    shifted_configs = torch.stack([torch.roll(configuration, shifts=int(s), dims=0) for s in shifts], dim=0)
    deltaxy_sqrd = (configuration - shifted_configs)**2
    r_sqrd = deltaxy_sqrd.view(N-1, -1, 2).sum(dim=2)
    sum_W = -torch.sum(torch.sum(torch.exp(-r_sqrd / (2 * a**2)), dim=1)) / 2
    return 1/N * sum_W

def BC(configuration):
    return configuration