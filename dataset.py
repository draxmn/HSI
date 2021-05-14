import torch.utils.data as data
import torch
import h5py

class DatasetFromHdf5(torch.utils.data.Dataset):

    def __init__(self, path):
        self.file_path = path
        self.dataset = None
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = file.get('data').shape[0]

    def __getitem__(self, index):
        if self.dataset is None:
            hf = h5py.File(self.file_path,'r')
            self.data = hf.get('data')
            self.target = hf.get('label')
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.target[index,:,:,:]).float()

    def __len__(self):
        return self.dataset_len