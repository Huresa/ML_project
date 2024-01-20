import torch

#XY model, d2n1 (2D vector on 1D lattice)
def XYd2n1(configuration, energy_parameters):
    J, mu, h, theta = energy_parameters
    """takes the (torch format) 1D lattice, the coupling constants and the magnetic field value (along theta)
    returns the total energy of the system"""
    h_coupling = - mu*h*torch.sum(torch.cos(configuration - theta)[:-1])
    J_interaction = - J*torch.sum((torch.cos(configuration - torch.roll(configuration, shifts = 1)))[:-1])
    return h_coupling + J_interaction

#boundary conditions (2D vector on 1D lattice)
def XYd2n1_BC(configuration):
    configuration = configuration%(2*torch.pi)
    copy_config = torch.zeros_like(configuration)
    copy_config[-1] = configuration[0]-configuration[-1]
    return configuration+copy_config

#XY model, d2n2 (2D vector on 2D lattice)
def XYd2n2(configuration, energy_parameters):
    J, mu, h, theta = energy_parameters
    """takes the (torch format) 2D lattice (as a 1D tensor),
    the coupling constants and the magnetic field value (along theta)
    returns the total energy of the system"""
    N = int(len(configuration) ** 0.5)
    lattice = configuration.view(N, N)
    h_coupling = - mu*h*torch.sum(torch.cos(lattice - theta))
    J_interaction_lines = - J*torch.sum(torch.cos(lattice - torch.roll(lattice, shifts=1, dims=1)))
    J_interaction_columns = - J*torch.sum(torch.cos(lattice - torch.roll(lattice, shifts=1, dims=0)))
    return h_coupling + J_interaction_lines + J_interaction_columns

#boundary conditions (2D vector on 2D lattice)
def XYd2n2_BC(configuration):
    N = int(len(configuration) ** 0.5)
    configuration = configuration%(2*torch.pi)
    copy_config = torch.zeros_like(configuration)
    copy_config[-N:] = configuration[:N]-configuration[-N:]
    copy_config[N-1::N] = configuration[::N]-configuration[N-1::N]
    return configuration+copy_config

#XY model, d3n2 (2D vector on 3D lattice)
def XYd2n3(configuration, energy_parameters):
    J, mu, h, theta = energy_parameters
    """takes the (torch format) 3D lattice (as a 1D tensor),
    the coupling constants and the magnetic field value (along theta)
    returns the total energy of the system"""
    N = int((len(configuration)+1) ** (1/3))
    lattice = configuration.view(N, N, N)
    h_coupling = - mu*h*torch.sum(torch.cos(lattice - theta))
    J_interaction_lines = - J*torch.sum(torch.cos(lattice - torch.roll(lattice, shifts=1, dims=2)))
    J_interaction_columns = - J*torch.sum(torch.cos(lattice - torch.roll(lattice, shifts=1, dims=1)))
    J_interaction_width = - J*torch.sum(torch.cos(lattice - torch.roll(lattice, shifts=1, dims=0)))
    return h_coupling + J_interaction_lines + J_interaction_columns + J_interaction_width

#boundary conditions (2D vector on 3D lattice)
def XYd2n3_BC(configuration):
    N = int((len(configuration)+1) ** (1/3))
    configuration = configuration%(2*torch.pi)
    copy_config = torch.zeros_like(configuration)
    copy_config[N-1::N] = configuration[::N]-configuration[N-1::N]
    copy_config[-N**2:] = configuration[:N**2]-configuration[-N**2:]
    for k in range(N):
        copy_config[(k+1)*N**2-N:(k+1)*N**2] = configuration[k*N**2:k*N**2+N]-configuration[(k+1)*N**2-N:(k+1)*N**2]
    return configuration + copy_config
