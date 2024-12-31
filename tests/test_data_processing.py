import os
import pytest
import torch
import pandas as pd
from torch.utils.data import DataLoader
from insightflow.data_processing import MyDataset  

csv_path = '/home/oliver/insightflow/tests/utils/sample_data.csv'

class TestMyDataset:
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.dataset = MyDataset(csv_path)
        yield

    def test_dataset_loading(self):
        dataset = self.dataset
        assert len(dataset) == 2
        
        x, y = dataset[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (15,)
        assert y.item() in [0, 1]

    def test_map_dataset(self):
        processed_data = self.dataset.processed_data

        assert 'x' in processed_data.columns
        assert 'y' in processed_data.columns
        assert len(processed_data) == 2
        
        assert processed_data['y'][0] == 1
        assert processed_data['y'][1] == 1

        assert processed_data['x'][0].shape == (15,)
        assert processed_data['x'][1].shape == (15,)

    def test_data_access(self):
        dataset = self.dataset
        
        assert len(dataset) == 2
        
        x, y = dataset[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (15,)
        assert y.item() in [0, 1]

    def test_tensor_format(self):
        x, y = self.dataset[0]
        
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_dataloader(self):
        dataloader = DataLoader(self.dataset, batch_size=1)

        for x, y in dataloader:
            assert isinstance(x, torch.Tensor)
            assert isinstance(y, torch.Tensor)
            assert x.shape == (1, 15)
            assert y.item() in [0, 1]
