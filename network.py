import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, input_dim, out_dim, hyperparam_dict):
        super(network, self).__init__()
        self.in_dim = input_dim
        self.out_dim = out_dim
        self.num_h_layers = int(hyperparam_dict["hidden_layers"])
        self.num_neurons = int(hyperparam_dict["neurons"])

        self.in_layer = nn.Linear(self.in_dim, self.num_neurons)
        self.out_layer = nn.Linear(self.num_neurons, self.out_dim)
        self.h_layers = nn.ModuleList(nn.Linear(self.num_neurons, self.num_neurons)
                                      for _ in range(self.num_h_layers-1))

    def activation(self, x):
        return torch.tanh(x)

    def forward(self, features):
        features = torch.tanh(self.in_layer(features))
        for affine in self.h_layers:
            features = self.activation(affine(features))
        output = self.out_layer(features)
        return output
