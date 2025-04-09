import torch

__all__ = ["torch_delete_rows"]


def torch_delete_rows(tensor, indices):
    indices_to_keep = torch.ones(tensor.shape[0], dtype=torch.bool)
    indices_to_keep[indices] = False
    return tensor[indices_to_keep]


def find_duplicates(x):
    _, inv, counts = torch.unique(x, return_inverse=True, return_counts=True, dim=0)
    indices_to_delete = [
        index
        for i, c in enumerate(counts)
        if c > 1
        for index in torch.where(inv == i)[0][1:].tolist()
    ]
    return indices_to_delete


def find_nan_rows(x):
    mask = torch.isnan(x).any(dim=1)
    indices_to_delete = mask.nonzero().flatten().tolist()
    return indices_to_delete
