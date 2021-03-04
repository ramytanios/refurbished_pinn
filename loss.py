import torch

L2_NORM = lambda data : torch.mean(data ** 2).item()

def relative_test_error(data_test, network):
    true_label = data_test[:, -1]
    features = data_test[:, 0:-1]
    pred_label = network(features)
    return L2_NORM(true_label - pred_label) / L2_NORM(true_label)

class Loss:
    def __init__(self, network, model):
        self.network = network
        self.model = model

    def get_pde_loss(self, data_interior):
        """ Compute the PDE residual loss """
        features = data_interior
        pde_residual = self.model.residual(features, self.network)
        return L2_NORM(pde_residual)

    def get_boundary_condition_loss(self, data_bc):
        """ Compute the boundary conditions loss """
        features = data_bc[:, 0:-1]
        true_label = data_bc[:, -1].reshape(-1,)
        pred_label = self.network(features).reshape(-1,)
        return L2_NORM(true_label - pred_label)

    def get_initial_condition_loss(self, data_ic):
        """ Compute the initial condition loss """
        features = data_ic[:, 0:-1]
        true_label = data_ic[:, -1].reshape(-1,)
        pred_label = self.model.network(features).reshape(-1,)
        return L2_NORM(true_label - pred_label)

    def get_measurements_loss(self, data_measurements):
        """" Computes the observed data loss """
        features = data_measurements[:, 0:-1] 
        true_label = data_measurements[:, -1]
        pred_label = self.network(features)
        return L2_NORM(true_label - pred_label)

    def get_regularization_loss(self, p):
        """ Computes the L_p regularization loss """
        lp_regularization = 0.0
        for param in self.network.parameters():
            lp_regularization = lp_regularization + torch.norm(param, p) ** p
        return lp_regularization.item()


