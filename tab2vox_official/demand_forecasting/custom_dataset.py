import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

class MND_Dataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 labels: pd.DataFrame,
                 key_features: list,
                 algo_type: str = 'voxel',
                 label_name = 'LABEL'):
        self.data = data
        self.labels = labels        
        self.algo_type = algo_type
        self.label_name = label_name
        self.key_features = key_features
        super().__init__()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.algo_type == 'voxel':

            x = self.data[idx,:,:,:,4:]          
            y = self.labels[idx]                 
            keys = self.data[idx,:,:,:,0:4]      
            
            return x, y, keys
