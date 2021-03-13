import torch
import os, sys
#sys.path.append(os.getcwd() + '/..')
#sys.path.append(os.getcwd())
import model
import dataset
import matplotlib.pyplot as plt

my_model = model.heat_model(1, 0.5)
interior_size = 50
boundary_size = 0
initial_size = 50
meas_size = 50
batch_size = 1 * (interior_size + boundary_size + initial_size + meas_size)
loaders = dataset.data_loaders(my_model,
                               int(batch_size),
                               interior_size,
                               boundary_size,
                               interior_size,
                               meas_size)

for (x, y, z, w) in zip(loaders[0], loaders[1], loaders[2], loaders[3]):
    print ("hello")
    plt.scatter(x[:, 0].numpy(), x[:, 1].numpy(), marker='o', label='interior')
    plt.scatter(y[:, 0].numpy(), y[:, 1].numpy(), marker='^', label='boundary')
    plt.scatter(z[:, 0].numpy(), z[:, 1].numpy(), marker='s', label='initial')
    plt.scatter(w[:, 0].numpy(), w[:, 1].numpy(), marker='+', label='measurements')
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.show()
