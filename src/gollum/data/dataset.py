from torch import Tensor
from torch.utils.data import Dataset


class SingleSampleDataset(Dataset):
    def __init__(self, x: Tensor, y: Tensor):
        self.x = x
        self.y = y
        self.samples = ((x, y),)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.samples[idx]


