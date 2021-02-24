import torch

def rescale(x, domain):
    for i in range (x.shape[1]):
        x[:, 0] = x[:, 0] * (domain[i][1] - domain[i][0]) + domain[i][0]
    return x

def assemble_dataset_interior(domain, size_interior):
    dim = domain.shape[0]
    return rescale(torch.rand(size_interior, dim), domain)

def assemble_dataset_initial(model, domain, size_initial):
    dim = domain.shape[0]
    x = rescale(torch.rand(size_initial, dim), domain)
    x[:, 0] = domain[0][0]
    return torch.cat((x, model.initial_condition(x)), 0)

def assemble_dataset_measurements(model, domain, size_measurements):
    dim = domain.shape[0]
    x = rescale(torch.rand(size_measurements, dim), domain)
    return torch.cat((x, model.exact(x)), 0)

def assemble_dataset_boundary(model, domain, size_boundary):
    dim = domain.shape[0]
    x = rescale(torch.rand(size_boundary, dim), domain)
    N = size_boundary / 2 / (dim - 1)
    for i in range(dim):
        idx = 2 * i - 1
        x[idx * N : (idx+1) * N] = domain[i][0]
        x[(idx+1) * N : (idx+2) * N] = domain[i][1]
    return torch.cat((x, model.exact(x)), 0)

def assemble_datasets(model, domain, *sizes):
    return [assemble_dataset_interior(domain, sizes[0]),
            assemble_dataset_boundary(model, domain, sizes[1]),
            assemble_dataset_initial(model, domain, sizes[2]),
            assemble_dataset_measurements(model, domain, sizes[3])]

class Dataset(torch.utils.data.Dataset):
    def __init__(self, model, *sizes):
        self.datasets = assemble_datasets(model, sizes[0], sizes[1], sizes[2], sizes[3])

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

#loader = torch.utils.data.DataLoader(concat_dataset, batch_size=batch_size, shuffle=True)
