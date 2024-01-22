import torch

def U(configuration, energy_parameters):
    a, L = energy_parameters
    N = configuration.shape[0] // 2
    shifts = torch.arange(2, 2 * N, 2)  # Create a tensor of shifts
    shifted_configs = torch.stack([torch.roll(configuration, shifts=int(s), dims=0) for s in shifts], dim=0)

    # Calculate delta x and y with periodic boundary conditions
    deltaxy = configuration - shifted_configs
    deltaxy = (deltaxy - L * torch.round(deltaxy / L))**2

    # Calculate squared distances
    r_sqrd = deltaxy.view(N-1, -1, 2).sum(dim=2)
    
    # Energy calculation
    sum_W = -torch.sum(torch.sum(torch.exp(L**2*(1-torch.cos(2*torch.pi*(r_sqrd**(1/2))/L)**2) / (4* torch.pi**2 * a**2)), dim=1))
    
    return 1/(2*N) * sum_W

def BC(configuration):
    L = 1
    return configuration%L