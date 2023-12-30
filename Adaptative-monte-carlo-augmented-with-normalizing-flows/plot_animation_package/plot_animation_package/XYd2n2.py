import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def trajectory(model_file, trajectory_id, k_max, k_lang, U, energy_parameters, array_of_model_configurations):
    
    N = int(len(array_of_model_configurations[0,0]) ** 0.5)
    lattice_memory = array_of_model_configurations[:, trajectory_id, :]
    
    plt.rcParams["animation.html"] = "jshtml"
    plt.rcParams['figure.dpi'] = 150
    plt.ioff()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.set_size_inches(8, 5)

    magnetisation = torch.mean(lattice_memory, dim=1, keepdim=True).numpy()
    std_magnetisation = torch.std(lattice_memory, dim=1, keepdim=True).numpy()

    energy_memory = []
    for elem in lattice_memory:
        energy_memory.append(U(elem, energy_parameters))

    theta = energy_parameters[-1]
    minimal_energy = U(torch.ones_like(array_of_model_configurations[0,0])*theta, energy_parameters)

    pos = ax1.imshow(lattice_memory[0].detach().view(N,N).numpy(), cmap='viridis', vmin  = 0, vmax = 2*torch.pi)

    def animate(k):
        ax1.cla()
        ax1.imshow(lattice_memory[k].detach().view(N,N).numpy(), cmap='viridis', vmin  = 0, vmax = 2*torch.pi)
        ax1.set_xlabel('Horizontal position [AU]')
        ax1.set_ylabel('Vertical position [AU]')
        ax1.set(yticklabels=[])
        ax1.set(xticklabels=[])

        ax2.cla()
        ax2.scatter(k, energy_memory[k])
        ax2.plot(energy_memory)
        ax2.set_xlabel("Number of MALA iterations")
        ax2.set_ylabel("Total energy of the system [AU]")
        ax2.axhline(y=minimal_energy, color = 'tab:orange')

        ax3.cla()
        ax3.scatter(k, magnetisation[k])
        ax3.plot(magnetisation)
        ax3.set_xlabel("Number of MALA iterations")
        ax3.set_ylabel("Magnetisation [AU]")
        ax3.axhline(y=theta, color = 'tab:orange')

        ax4.cla()
        ax4.scatter(k, std_magnetisation[k])
        ax4.plot(std_magnetisation)
        ax4.set_xlabel("Number of MALA iterations")
        ax4.set_ylabel("Std magnetisation [AU]")

    fig.colorbar(pos, ax=ax1, label='Node spin angle [rad]', orientation='horizontal')

    plt.tight_layout()
    animation = FuncAnimation(fig, animate, frames=k_max)
    
    animation.save(model_file+'.gif', writer='imagemagick')

    return None