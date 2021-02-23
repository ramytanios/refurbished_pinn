import torch

def L2_NORM(data):
    return torch.mean(data ** 2).item()

def relative_test_error(data_test, network):
    true_label = data_test[:, -1]
    features = data_test[:, 0:-1]
    pred_label = network(features)
    return L2_NORM(true_label - pred_label) / L2_NORM(true_label)

class loss:
    def __init__(self, network, model):
        self.network = network
        self.model = model

    def get_measurements_loss(self, data_measurements):
        true_label = data_measurements[:, -1].reshape(-1,)
        features = data_measurements[:, 0:-1]
        pred_label = self.network(features).reshape(-1,)
        return L2_NORM(true_label - pred_label)

    def get_regularization_loss(self, p):
        lp_regularization = 0.0
        for param in self.network.parameters():
            lp_regularization = lp_regularization + torch.norm(param, p) ** p
        return lp_regularization.item()

    def get_pde_loss(self, data_internal):
        features = data_internal
        pde_residual = self.model.residual(features)
        return L2_NORM(pde_residual)

    def get_boundary_condition_loss(self, data_bc):
        features = data_bc[:, 0:-1]
        true_label = self.model.boundary_condition(features)
        pred_label = self.network(features)
        return L2_NORM(true_label - pred_label)

    def get_initial_condition_loss(self, data_ic):
        features = data_ic[:, 0:-1]
        true_label = data_ic[:, -1]
        pred_label = self.model.initial_condition(features)
        return L2_NORM(true_label - pred_label)


