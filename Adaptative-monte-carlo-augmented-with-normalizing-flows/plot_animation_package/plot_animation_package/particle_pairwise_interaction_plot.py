import torch
import matplotlib.pyplot as plt
from os.path import dirname 
from matplotlib.animation import FuncAnimation

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
    
    return n, k_max, k_lang, energy_parameters, array_of_model_configurations.detach() 


def animation(model_file, trajectory_id, history):
    n, k_max, k_lang, energy_parameters, array_of_model_configurations = read(model_file)
    
    x_positions = array_of_model_configurations[:,trajectory_id,0::2].numpy()
    y_positions = array_of_model_configurations[:,trajectory_id,1::2].numpy()

    fig, ax = plt.subplots()
    plt.ioff()
    scat = ax.scatter([], [], marker='o', color='tab:blue')

    def init():
        ax.set_xlim(min(map(min, x_positions)), max(map(max, x_positions)))
        ax.set_ylim(min(map(min, y_positions)), max(map(max, y_positions)))
        return scat,

    def update(frame):
        x, y = x_positions[frame], y_positions[frame]
        scat.set_offsets(list(zip(x, y)))
        return scat,

    anim = FuncAnimation(fig, update, frames=k_max, init_func=init, blit=True)
    anim.save('hello.gif', writer='imagemagick')

    return None