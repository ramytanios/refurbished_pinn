import torch
from torch import nn

class network(nn.Module):
    def __init__(self, input_dim, out_dim, hyperparam_dict):
        super(network, self).__init__()
        self.in_dim = input_dim
        self.out_dim = out_dim
        self.net_arch = hyperparam_dict
        self.num_h_layers = int(hyperparam_dict["hidden_layers"])
        self.num_neurons = int(hyperparam_dict["neurons"])

        self.in_layer = nn.Linear(self.in_dim, self.num_neurons)
        self.out_layer = nn.Linear(self.num_neurons, self.out_dim)
        self.h_layers = nn.ModuleList(nn.Linear(self.num_neurons, self.num_neurons)
                                      for _ in range(self.num_h_layers-1))

        self.batch_in = nn.BatchNorm1d(self.num_neurons)
        self.batch = nn.ModuleList(nn.BatchNorm1d(self.num_neurons)
                                   for _ in range(self.num_h_layers-1))

    def forward(self, inp):
        inp = torch.tanh(self.batch_in(self.in_layer(inp)))
        for l, b in zip(self.h_layers, self.batch):
            inp = torch.tanh(b(l(inp)))
        inp = self.out_layer(inp)
        return inp