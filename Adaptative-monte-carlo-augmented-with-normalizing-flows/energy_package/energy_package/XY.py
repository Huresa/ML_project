import torch

#XY model, d2n1 (2D vector on 1D lattice)
def XYd2n1(configuration, energy_parameters):
    J, mu, h, theta = energy_parameters
    """takes the (torch format) 1D lattice, the coupling constants and the magnetic field value (along theta)
    returns the total energy of the system"""
    h_coupling = - mu*h*torch.sum(torch.cos(configuration - theta)[:-1])
    J_interaction = - J*torch.sum((torch.cos(configuration - torch.roll(configuration, shifts = 1)))[:-1])
    return h_coupling + J_interaction

#boundary conditions
def XYd2n1_BC(configuration):
    configuration = configuration%(2*torch.pi)
    copy_config = torch.zeros_like(configuration)
    copy_config[-1] = configuration[0]-configuration[-1]
    return configuration+copy_config

#XY model, d2n2 (2D vector on 2D lattice)
def XYd2n2(lattice, energy_parameters):
    J, mu, h, theta = energy_parameters
    """takes the (torch format) 1D lattice, the coupling constants and the magnetic field value (along theta)
    returns the total energy of the system"""
    h_coupling = - mu*h*torch.sum(torch.cos(lattice - theta)[:-1,:-1])
    J_interaction_lines = - J*torch.sum((torch.cos(lattice - torch.roll(lattice, shifts = (0,1), dims=(0, 1))))[:-1,:-1])
    J_interaction_columns = - J*torch.sum((torch.cos(lattice - torch.roll(lattice, shifts = (1,0), dims=(0, 1))))[:-1,:-1])
    return h_coupling + J_interaction_lines + J_interaction_columns

#boundary conditions
def XYd2n2_BC(configuration):
    configuration = configuration%(2*torch.pi)
    copy_config = torch.zeros_like(configuration)
    copy_config[-1,:] = configuration[0,:]
    copy_config[:,-1] = configuration[:,0]
    return configuration+copy_config