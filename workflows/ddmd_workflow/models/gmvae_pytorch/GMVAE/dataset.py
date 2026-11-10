import numpy as np
import torch
from torch.utils.data import Dataset as PyTorchDataset, DataLoader

def one_hot(labels, depth):
    one_hot = np.zeros((len(labels), depth))
    labels = np.reshape(labels, (len(labels, )))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

class Trainset(object):
    def __init__(self):
        self.data = None
        self.labels = None
        self.current_index = 0

    def next_batch(self, size):
        assert size <= len(self.data), 'train set is too small!'
        batch = self.data[self.current_index:self.current_index+size]
        self.current_index += size
        self.current_index = self.current_index % len(self.data)
        return batch

class Testset(object):
    def __init__(self):
        self.data = None
        self.labels = None

class CustomDataset(object):
    def __init__(self, k):
        self.k = k
        self.train = Trainset()
        self.test = Testset()

    def setTrainData(self, data, labels=None):
        self.train.data = data
        if isinstance(labels, np.ndarray):
            self.train.labels = one_hot(labels, self.k)

    def setTestData(self, data, labels=None):
        self.test.data = data
        if isinstance(labels, np.ndarray):
            self.test.labels = one_hot(labels, self.k)

# Standard PyTorch Dataset Wrapper
class TorchDataset(PyTorchDataset):
    def __init__(self, data_array):
        self.data = torch.tensor(data_array, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_and_mix_data_nolabel(path=None, k=3, data=None, test_ratio=0.0, randomize=True, batch_size=None):
    if data is None and path is not None:
        data = np.load(path)
    elif data is None:
        # Dummy data fallback for testing if no path/data provided
        data = np.random.randn(1000, 784).astype(np.float32)

    m = data.shape[0]

    if randomize:
        indices = np.random.permutation(m)
        data = data[indices, :]

    train_ratio = 1.0 - test_ratio
    train_data = data[:int(m * train_ratio)]
    test_data = data[int(m * train_ratio):]

    # Return PyTorch DataLoader if batch_size is provided
    if batch_size is not None:
        torch_ds = TorchDataset(train_data)
        return DataLoader(torch_ds, batch_size=batch_size, shuffle=True)

    dataset = CustomDataset(k)
    dataset.setTrainData(train_data)
    dataset.setTestData(test_data)
    return dataset
