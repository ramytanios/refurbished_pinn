import torch

class heat_model:
    def __init__(self, param, spatial_dimension):
        self.param = param
        self.d = spatial_dimension

    def residual(self, x, network):
        x.requires_grad = True
        Id_tensor = torch.ones(x.shape[0], )
        if torch.cuda.is_available():
            Id_tensor = Id_tensor.cuda()

        u = network(x).reshape(-1, )
        grad_u = torch.autograd.grad(u, x, grad_outputs=Id_tensor, create_graph=True)[0]
        grad_t_u = grad_u[:, 0]
        residual = grad_t_u
        for i in range(self.d):
            grad_xdxd_u = torch.autograd.grad(grad_u[:, self.d], x,
                                              grad_outputs=Id_tensor, create_graph=True)[0][:, self.d]
            residual = residual - self.param  * grad_xdxd_u
        return residual

    def initial_condition(self, x):
        return torch.mean(x ** 2, dim=1).reshape(-1, 1)

    def exact_solution(self, x, t):
        return self.initial_condition(x) + 2 * t

    def boundary_condition(self, x, t):
        return self.exact_solution(x, t)


