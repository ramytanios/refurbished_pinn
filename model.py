import torch

class heat_model:
    def __init__(self, spatial_dimension, *param):
        """ Constructor assembling the domain and some PDE parameters """
        self.param = param
        self.d = spatial_dimension
        self.domain = torch.tensor(()).new_zeros(size=(spatial_dimension + 1, 2))
        self.T = 1.
        self.domain[0] = torch.tensor([0, self.T])
        for i in range(1, spatial_dimension + 1):
            self.domain[i] = torch.tensor([0, 1])

    def residual(self, x, network):
        """ Input feature x = (t,x) and the network u_{theta} and returns the PDE residual """
        x.requires_grad = True
        Id_tensor = torch.ones(x.shape[0], )
        if torch.cuda.is_available():
            Id_tensor = Id_tensor.cuda()

        u = network(x).reshape(-1, )
        grad_u = torch.autograd.grad(u, x, grad_outputs=Id_tensor, create_graph=True)[0]
        grad_t_u = grad_u[:, 0]
        residual = grad_t_u
        for i in range(self.d):
            grad_xdxd_u = torch.autograd.grad(grad_u[:, self.d],
                                              x,
                                              grad_outputs=Id_tensor,
                                              create_graph=True)[0][:, self.d]
            residual = residual - self.param  * grad_xdxd_u
        return residual

    def initial_condition(self, x):
        """ Input spatial dimension x and returns the initial condition at x """
        return torch.mean(x ** 2, dim=1).reshape(-1, 1)

    def exact_solution(self, x):
        """ Input feature x = (t,x) and returns the exact solution at x """ 
        return self.initial_condition(x) + 2 * x[:, 0].reshape(-1, 1)

    def boundary_condition(self, x):
        """ Input feature x = (t,x) and returns the boundary condition at x """
        return self.exact_solution(x)
