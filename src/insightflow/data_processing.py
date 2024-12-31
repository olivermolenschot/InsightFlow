from torch.utils.data import Dataset
from pandas import DataFrame
import pandas as pd
import numpy as np
import torch

class MyDataset(Dataset):

    COLUMNS = ['Open','High','Low','Close','Volume']
    def __init__(self, 
                csv_path: str
                ):
        self.data = pd.read_csv(csv_path)
        self.processed_data = self.map_dataset(self.data)

    def map_dataset(self, data: pd.DataFrame):
        processed_data = {'x': [], 'y': []}
        
        for idx in range(3, len(data)):
            x = []
            for j in range(idx-3, idx):
                x.append(data.iloc[j][self.COLUMNS].values)  
            x = np.concatenate(x) 
            
            if data.iloc[idx]['Close'] > data.iloc[idx-1]['Close']:
                y = 1
            else:
                y = 0
            
            processed_data['x'].append(x)
            processed_data['y'].append(y)
        
        processed_df = pd.DataFrame(processed_data)
        
        return processed_df
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.processed_data['x'].iloc[idx], dtype=torch.float32)
        y = torch.tensor(self.processed_data['y'].iloc[idx], dtype=torch.float32)
        return x, y