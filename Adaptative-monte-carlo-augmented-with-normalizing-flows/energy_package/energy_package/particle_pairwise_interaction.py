import torch


#2D box, small particles compared to L size of the box (a<<L)
def W(r, energy_parameters):
    a = energy_parameters[0]
    return -torch.exp(-(r**2)/(2*a**2))

def U(configuration, energy_parameters):
    a, L = energy_parameters
    N = int(len(configuration)/2)

    sum = 1/(2*N) * 