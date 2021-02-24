import torch
import model
import dataset
import matplotlib.pyplot as plt

my_model = model.heat_model(1, 0.5)
dataset = dataset.Dataset(my_model, 100, 100, 100, 5)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
for x in loader:
    print (x[1])
    plt.scatter(x[0][:, 0].numpy(), x[0][:, 1].numpy(), marker='o', label='interior')
    plt.scatter(x[1][:, 0].numpy(), x[1][:, 1].numpy(), marker='^', label='boundary')
    plt.scatter(x[2][:, 0].numpy(), x[2][:, 1].numpy(), marker='s', label='initial')
    plt.scatter(x[3][:, 0].numpy(), x[3][:, 1].numpy(), marker='+', label='measurements')
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.show()
