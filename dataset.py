import torch
import numpy as np

def rescale(x, domain):
    """ Input x ~ U(0,1)^{d+1} and returns x in domain """
    for i in range (x.shape[1]):
        x[:, 0] = x[:, 0] * (domain[i][1] - domain[i][0]) + domain[i][0]
    return x

def assemble_dataset_interior(domain, size_interior):
    """ Input PDE model and size and returns the interior points dataset """ 
    """ Unlabeled dataset as torch tensor """ 
    dim = domain.shape[0]
    return rescale(torch.rand(size_interior, dim), domain)

def assemble_dataset_initial(model, size_initial):
    """ Input PDE model and size and returns the initial time points dataset """
    """ Labeled dataset as torch tensor """
    dim = model.domain.shape[0]
    x = rescale(torch.rand(size_initial, dim), model.domain)
    x[:, 0] = model.domain[0][0]
    return torch.cat((x, model.initial_condition(x)), 1)

def assemble_dataset_measurements(model, size_measurements):
    """ Input PDE model and size and returns the measurements points dataset """
    """ Labeled dataset as torch tensor """
    dim = model.domain.shape[0]
    x = rescale(torch.rand(size_measurements, dim), model.domain)
    return torch.cat((x, model.exact_solution(x)), 1)

def assemble_dataset_boundary(model, size_boundary):
    """ Input PDE model and size and returns the boundary points dataset """
    """ Labeled dataset as torch tensor """
    dim = model.domain.shape[0]
    x = rescale(torch.rand(size_boundary, dim), model.domain)
    N = int(size_boundary / 2 / (dim - 1))
    for i in range(dim - 1):
        idx = int(2 * i)
        x[idx * N : (idx+1) * N, i + 1] = model.domain[i][0]
        x[(idx+1) * N : (idx+2) * N, i + 1] = model.domain[i][1]
    return torch.cat((x, model.exact_solution(x)), 1)

class Dataset(torch.utils.data.Dataset):
    """ Dataset class """
    def __init__(self, assemble_dataset, model, size):
        """ Constructor that assemble the dataset """
        self.datasets = assemble_dataset(model, size)

    def __getitem__(self, i):
        """ Input index i and return the sample at i """
        return self.datasets[i]

    def __len__(self):
        """ Returns the number of samples in the dataset """
        return self.datasets.shape[0]

def data_loaders(model, batch_size, *sizes):
    """ Inputs model PDE, batch_size and training sets sizes and returns 
        a dataloader for each dataset """
    dataset_interior = Dataset(assemble_dataset_interior, model.domain, sizes[0])
    dataset_boundary = Dataset(assemble_dataset_boundary, model, sizes[1])
    dataset_initial = Dataset(assemble_dataset_initial, model, sizes[2])
    dataset_measurements = Dataset(assemble_dataset_measurements, model, sizes[3])
    list_of_dataset = [dataset_interior, dataset_boundary, dataset_initial, dataset_measurements]

    def get_len(dataset):
        return len(dataset)
    N = np.array(list(map(get_len, list_of_dataset)))
    N_total = sum(N)
    batch_size_of_each = (N / N_total * batch_size)
    loaders = []
    for i in range(4):
        loaders.append(torch.utils.data.DataLoader(list_of_dataset[i],
                                                   batch_size = int(batch_size_of_each[i]),
                                                   shuffle = True))
    return loaders

