import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from os.path import dirname 

def read(model_file):
    path = dirname(model_file)
    array_of_model_configurations = torch.load(path+'\\array_of_model_configurations.pt')

    parameters_file = path+"\\parameters.txt"

    parameters = {}
    with open(parameters_file, 'r') as file:
        for line in file:
            key, value = line.strip().split('\t')
            if value.startswith('[') and value.endswith(']'):
                value = [float(x.strip()) for x in value[1:-1].split(',')]
            elif key == 'beta' or key == 'time_step' or key == 'epsilon':
                value = float(value)
            else:
                value = int(value)
            parameters[key] = value

    print(parameters)
    for key, value in parameters.items():
        globals()[key] = value
    
    return k_max, k_lang, energy_parameters, array_of_model_configurations.detach() 

def simple_plot(model_file, trajectory_id, U):
    
    k_max, k_lang, energy_parameters, array_of_model_configurations = read(model_file)

    N = int((len(array_of_model_configurations[0,0])+1) ** (1/3))
    lattice_memory = array_of_model_configurations
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.set_size_inches(8, 5)
    theta = energy_parameters[-1]
    
    magnetisation_sin = torch.mean(torch.mean(torch.sin(lattice_memory-torch.tensor(theta)), dim=2, keepdim=True), dim=1, keepdim=True).numpy().squeeze()
    magnetisation_cos = torch.mean(torch.mean(torch.cos(lattice_memory-torch.tensor(theta)), dim=2, keepdim=True), dim=1, keepdim=True).numpy().squeeze()
    std_magnetisation_sin = torch.mean(torch.std(torch.sin(lattice_memory-torch.tensor(theta)), dim=2, keepdim=True), dim=1, keepdim=True).numpy().squeeze()
    std_magnetisation_cos = torch.mean(torch.std(torch.cos(lattice_memory-torch.tensor(theta)), dim=2, keepdim=True), dim=1, keepdim=True).numpy().squeeze()

    energy_memory = []
    for elem in lattice_memory[:, trajectory_id, :]:
        energy_memory.append(U(elem, energy_parameters))

    minimal_energy = U(torch.ones_like(array_of_model_configurations[0,0])*theta, energy_parameters)

    pos = ax1.imshow(lattice_memory[-1,0,:].detach().view(N,N,N)[0].numpy(), cmap='viridis', vmin  = 0, vmax = 2*torch.pi)
    ax1.set_xlabel('Horizontal position [AU]')
    ax1.set_ylabel('Vertical position [AU]')
    ax1.set(yticklabels=[])
    ax1.set(xticklabels=[])

    ax2.plot(energy_memory)
    ax2.set_xlabel("Number of MALA iterations")
    ax2.set_ylabel("Total energy of the system [AU]")
    ax2.axhline(y=minimal_energy, color = 'tab:orange')


    ax3.plot(magnetisation_cos, label='mean(cos(alpha-theta)')
    ax3.plot(magnetisation_sin, label='mean(sin(alpha-theta)')
    ax3.set_xlabel("Number of MALA iterations")
    ax3.set_ylabel("Magnetisation [AU]")
    ax3.set_ylim(-1,1)
    ax3.legend()

    ax4.plot(std_magnetisation_cos, label='std(cos(alpha-theta))')
    ax4.plot(std_magnetisation_sin, label='std(sin(alpha-theta))')
    ax4.set_xlabel("Number of MALA iterations")
    ax4.set_ylabel("Std magnetisation [AU]")
    ax4.legend()

    fig.colorbar(pos, ax=ax1, label='Node spin angle [rad]', orientation='horizontal')
    plt.tight_layout()
    plt.show()

    fig.savefig(model_file+'.png')

    return None