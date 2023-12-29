import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import pi


def trajectory(model_file, trajectory_id, k_max, k_lang, U, energy_parameters, array_of_model_configurations):
    lattice_memory = array_of_model_configurations[:, trajectory_id, :]

    plt.rcParams["animation.html"] = "jshtml"
    plt.rcParams['figure.dpi'] = 300
    plt.ioff()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.set_size_inches(8, 5)
    
    N = len(lattice_memory[0])
    nodes = [_ for _ in range(N)]

    energy_memory = []


    magnetisation = torch.mean(lattice_memory, dim=1, keepdim=True).numpy()
    std_magnetisation = torch.std(lattice_memory, dim=1, keepdim=True).numpy()

    for elem in lattice_memory:
        energy_memory.append(U(elem, energy_parameters))

    theta = energy_parameters[-1]
    minimal_energy = U(torch.ones(N)*theta, energy_parameters)

    def animate(k):
        ax1.cla()
        ax1.plot(nodes, lattice_memory[k].detach().numpy(), linestyle = '', marker = 'o')
        ax1.set_xlim(0,N)
        ax1.set_ylim(0, 2*pi)
        ax1.set_xlabel("Nodes")
        ax1.set_ylabel("Angle of the node's spin [rad]")

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

    plt.tight_layout()
    animation = FuncAnimation(fig, animate, frames=k_max)
    
    animation.save(model_file+'.gif', writer='imagemagick')

    return None


def plot(beta, U, energy_parameters, model_file, renormalization_flow, base_distribution):

    flow = renormalization_flow
    flow.load_state_dict(torch.load(model_file))
    flow.eval()

    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12,5))

    xlim = [-15,15]
    ylim = xlim
    x = torch.linspace(*xlim,200)
    y = torch.linspace(*ylim,200)

    xx, yy = torch.meshgrid(x,y)

    def log_rho_hat(x):
        return base_distribution.log_prob((flow.inverse(x))[0]) + flow.inverse(x)[1]

    z_flow = torch.exp(log_rho_hat(torch.stack([xx, yy], dim=-1).reshape(-1, 2)).detach()).reshape(xx.shape)
    ax1.pcolormesh(xx.numpy(),yy.numpy(),z_flow.numpy())
    ax1.set_title('Flow distribution')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    z_target = torch.exp(-beta*U(torch.stack([xx, yy]), energy_parameters))
    ax2.pcolormesh(xx.numpy(),yy.numpy(),z_target.numpy())
    ax2.set_title('Target distribution')
    ax2.set_xlabel('x')

    fig.savefig(model_file+'.png')

    return None