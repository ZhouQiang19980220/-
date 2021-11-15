import torch
import numpy as np
from torch.utils.data import Dataset

class MyDataSet(Dataset):
    def __init__(self, data_path, target_path):
        data = np.loadtxt(data_path)
        target = np.loadtxt(target_path)
        
        self.data = torch.FloatTensor(data)
        self.target = torch.FloatTensor(target)
    
            
    
    
    def __getitem__(self, index):
        return self.data[index], self.target[index]
    
    def __len__(self):
        return self.data.size()[0]