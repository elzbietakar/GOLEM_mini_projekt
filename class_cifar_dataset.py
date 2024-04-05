from torch.utils.data import Dataset
import numpy as np

class CifarDataset(Dataset):

    def __init__(self, labels, data: np.ndarray, transform=None):
        self.labels = labels
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            data = self.transform(data[idx])
        return data, label