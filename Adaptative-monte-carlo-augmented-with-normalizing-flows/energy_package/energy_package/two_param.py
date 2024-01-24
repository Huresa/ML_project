import torch

def bistable_circle(configuration, energy_parameters):
    '''energy of the system for a given configuration'''
    x, y = configuration[0], configuration[1]

    norm = torch.sqrt(x ** 2 + y ** 2)
    exp1 = torch.exp(-0.5 * ((x - 4) / 0.8) ** 2)
    exp2 = torch.exp(-0.5 * ((x + 4) / 0.8) ** 2)
    
    return 0.5 * ((norm - 5) / 0.4) ** 2 - torch.log(exp1 + exp2)

#boundary conditions
def bistable_circle_BC(configuration):
    return (configuration+torch.tensor(10))%20-10