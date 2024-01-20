import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def trajectory(model_file, trajectory_id, k_max, k_lang, U, energy_parameters, array_of_model_configurations):
    
    N = int((len(array_of_model_configurations[0,0])+1) ** (1/3))
    lattice_memory = array_of_model_configurations
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.set_size_inches(8, 5)
    theta = energy_parameters[-1]
    
    magnetisation = torch.mean(torch.mean(torch.cos(lattice_memory-torch.tensor(theta)), dim=2, keepdim=True), dim=1, keepdim=True).numpy().squeeze()
    std_magnetisation = torch.mean(torch.std(torch.cos(lattice_memory-torch.tensor(theta)), dim=2, keepdim=True), dim=1, keepdim=True).numpy().squeeze()

    energy_memory = []
    for elem in lattice_memory[:, trajectory_id, :]:
        energy_memory.append(U(elem, energy_parameters))

    minimal_energy = U(torch.ones_like(array_of_model_configurations[0,0])*theta, energy_parameters)

    ax2.plot(energy_memory)
    ax2.set_xlabel("Number of MALA iterations")
    ax2.set_ylabel("Total energy of the system [AU]")
    ax2.axhline(y=minimal_energy, color = 'tab:orange')

    ax3.plot(magnetisation)
    ax3.set_xlabel("Number of MALA iterations")
    ax3.set_ylabel("Deviation to theta [AU]")

    ax4.plot(std_magnetisation)
    ax4.set_xlabel("Number of MALA iterations")
    ax4.set_ylabel("Std magnetisation [AU]")

    plt.tight_layout()
    
    plt.savefig(model_file+'.png', dpi='1200')

    return None