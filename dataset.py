import torch

def rescale(x, domain):
    for i in range (x.shape[1]):
        x[:, 0] = x[:, 0] * (domain[i][1] - domain[i][0]) + domain[i][0]
    return x

def assemble_dataset_interior(model, size_interior):
    dim = model.domain.shape[0]
    return rescale(torch.rand(size_interior, dim), model.domain)

def assemble_dataset_initial(model, size_initial):
    dim = model.domain.shape[0]
    x = rescale(torch.rand(size_initial, dim), model.domain)
    x[:, 0] = model.domain[0][0]
    return torch.cat((x, model.initial_condition(x)), 1)

def assemble_dataset_measurements(model, size_measurements):
    dim = model.domain.shape[0]
    x = rescale(torch.rand(size_measurements, dim), model.domain)
    return torch.cat((x, model.exact_solution(x)), 1)

def assemble_dataset_boundary(model, size_boundary):
    dim = model.domain.shape[0]
    x = rescale(torch.rand(size_boundary, dim), model.domain)
    N = int(size_boundary / 2 / (dim - 1))
    for i in range(dim - 1):
        idx = int(2 * i)
        x[idx * N : (idx+1) * N, i + 1] = model.domain[i][0]
        x[(idx+1) * N : (idx+2) * N, i + 1] = model.domain[i][1]
    return torch.cat((x, model.exact_solution(x)), 1)

def assemble_datasets(model, *sizes):
    if sizes[3] == 0:
        return [assemble_dataset_interior(model, sizes[0]),
                assemble_dataset_boundary(model, sizes[1]),
                assemble_dataset_initial(model, sizes[2])]
    elif sizes[2] == 0:
        return [assemble_dataset_interior(model, sizes[0]),
                assemble_dataset_initial(model, sizes[2]),
                assemble_dataset_measurements(model, sizes[3])]
    else:
        return [assemble_dataset_interior(model, sizes[0]),
                assemble_dataset_boundary(model, sizes[1]),
                assemble_dataset_initial(model, sizes[2]),
                assemble_dataset_measurements(model, sizes[3])]

class Dataset(torch.utils.data.Dataset):
    def __init__(self, model, *sizes):
        self.datasets = assemble_datasets(model, sizes[0], sizes[1], sizes[2], sizes[3])

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return max(len(d) for d in self.datasets)
