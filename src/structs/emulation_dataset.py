import torch
from torch.utils.data import Dataset


class EmulationDataset(Dataset):
    """Minimal (x, y) dataset for PyTorch."""

    def __init__(self, x, y):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
