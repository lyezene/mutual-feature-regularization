import numpy as np
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    def __init__(self, data, normalize=True):
        self.data = data
        if normalize:
            self.normalize_data()

    def __getitem__(self, index):
        return self.data[index].astype(np.float32)

    def __len__(self):
        return len(self.data)

    def normalize_data(self):
        mean = self.data.mean(axis=(1, 2), keepdims=True)
        std = self.data.std(axis=(1, 2), keepdims=True)
        self.data = (self.data - mean) / (std + 1e-8)
