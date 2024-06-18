import torch
import numpy as np
import os
import re

from h5py import Dataset


def read_data(file_path):
    """Read data from a single file."""
    data = np.loadtxt(file_path, delimiter=',')
    return data

def read_data1(file_path):
    """Read data from a single file."""
    data = np.loadtxt(file_path, delimiter=' ')
    return data

'''


def read_data(file_path):
    return np.loadtxt(file_path, delimiter=',')
'''


def sort_files_numerically(files):
    """Sort files numerically based on their names."""
    return sorted(files, key=lambda x: int(re.search(r'\d+', x).group()))

def load_data_from_directory(directory_path,needsorted):
    """Load text files in numerical order and convert data to tensors."""
    data_list = []
    files = [f for f in os.listdir(directory_path) if f.endswith(".txt")]
    if needsorted == True:
        sorted_files = sort_files_numerically(files)
        for filename in sorted_files:
            file_path = os.path.join(directory_path, filename)
            data = read_data(file_path)
            #tensor = torch.tensor(data, dtype=torch.float32)
            data_list.append(data)
        return data_list
    else:
        for filename in files:
            file_path = os.path.join(directory_path,filename)
            data = read_data(file_path)
            #tensor = torch.tensor(data, dtype=torch.float32)
            data_list.append(data)
        return  data_list


def load_data_from_directory1(directory_path,needsorted):
    """Load text files in numerical order and convert data to tensors."""
    data_list = []
    files = [f for f in os.listdir(directory_path) if f.endswith(".txt")]
    if needsorted == True:
        sorted_files = sort_files_numerically(files)
        for filename in sorted_files:
            file_path = os.path.join(directory_path, filename)
            data = read_data1(file_path)
            #tensor = torch.tensor(data, dtype=torch.float32)
            data_list.append(data)
        return data_list
    else:
        for filename in files:
            file_path = os.path.join(directory_path,filename)
            data = read_data1(file_path)
            #tensor = torch.tensor(data, dtype=torch.float32)
            data_list.append(data)
        return  data_list


def split_data(data_tensors):
    """Split the data into training and testing datasets."""
    full_data = torch.cat(data_tensors, dim=0)
    total_samples = full_data.shape[0]
    train_size = int(total_samples * 0.8)
    test_size = total_samples - train_size
    train_data, val_data = torch.utils.data.random_split(full_data, [train_size, test_size])
    return train_data, val_data

def repeat_tensor_elements(tensor, times):
    new_tensor = [tensor]*times
    return new_tensor

class CustomDataset(Dataset):
    def __init__(self, input_data, coord_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.coord_data = coord_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        return self.input_data[index], self.coord_data[index], self.output_data[index]
