import numpy as np
import matplotlib.pyplot as plt
import random

final_data = np.array([])
count = 0

def generate_line(direction, num_points, noise=1):
    global final_data
    data = np.sort(np.random.normal(0, noise, num_points))
    if direction == 'down':
        data = data[::-1]
    data = data + final_data[-1] if len(final_data) > 0 else data
    final_data = np.concatenate((final_data, data))
    return data

def generate_data():
    global final_data
    final_data = np.array([])

    # three top
    # generate_line('up', 8, 1)  # Left shoulder
    # generate_line('down', 6, 1)  # Neckline
    # generate_line('up', 1, 2.5)  # Head
    # generate_line('down', 6, 2.5)  # Neckline
    # generate_line('up', 6, 1)  # Right shoulder
    # generate_line('down', 8, 1)  # Break neckline

    # double top 
    generate_line('up', 6, 1)  # Left shoulder
    generate_line('down', 6, 1)  # Neckline
    generate_line('up', 6, 1)  # Right shoulder
    generate_line('down', 6, 1)  # Break neckline

    final_data = final_data * -1
    return final_data

plt.ion()
for i in range(170):
    plt.clf()
    plt.plot(generate_data(), linewidth=random.randint(4,15),color='black')
    plt.xticks([])
    plt.yticks([])
    # 枠線を非表示にする
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.draw()
    plt.savefig('random_graph/graph'+str(count)+'.png')
    count += 1
    plt.pause(0.01)