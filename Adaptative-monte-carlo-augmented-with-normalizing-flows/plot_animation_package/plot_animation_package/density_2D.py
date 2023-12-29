import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D


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

def animation(model_file, n, k_max, array_of_model_configurations, history):
    
    positions = array_of_model_configurations
    values = history

    # Set up the figure and axis
    fig, ax = plt.subplots()

    # Store trails for each particle
    trail_length = 10  # You can adjust the trail length
    trails = torch.zeros((n, trail_length, 2))
    trail_values = torch.zeros((n, trail_length))

    # Turn off interactive mode
    plt.ioff()

    # Create scatter plot for particles
    particle_scatter = ax.scatter([], [], marker='o', color='blue')

    # Create line plot for trails
    trail_lines = [ax.add_line(Line2D([], [], color='orange', alpha=0.5)) for _ in range(n)]

    # Function to initialize the plot
    def init():
        ax.set_xlim(-10, 10)  # Adjust the limits based on your data
        ax.set_ylim(-10, 10)
        return particle_scatter, *trail_lines

    # Function to update the plot for each frame
    def update(frame):
        # Update trails
        trails[:, 1:, :] = trails[:, :-1, :]
        trails[:, 0, :] = positions[frame, :, :]
        trail_values[:, 1:] = trail_values[:, :-1]
        trail_values[:, 0] = values[frame, :]

        # Flatten the trails for plotting
        trail_data = trails.view(-1, 2).numpy()
        trail_color = trail_values.view(-1).numpy()

        # Update particle scatter
        particle_scatter.set_offsets(positions[frame].view(-1, 2).numpy())

        # Update trail lines and set different color for values 2 and 3
        for i in range(n):
            trail_lines[i].set_xdata(trail_data[i * trail_length:(i + 1) * trail_length, 0])
            trail_lines[i].set_ydata(trail_data[i * trail_length:(i + 1) * trail_length, 1])
            trail_lines[i].set_color('orange' if trail_color[i * trail_length] in [0, 1] else 'red')

        return particle_scatter, *trail_lines

    # Create the animation with a higher interval value for slower animation
    animation = FuncAnimation(fig, update, frames=k_max, init_func=init, blit=True, interval=100)

    # Save the animation (optional)
    animation.save(model_file+'.gif', writer='imagemagick', fps=10)
