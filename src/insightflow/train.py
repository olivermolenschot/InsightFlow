from data_processing import MyDataset
from trainer import Trainer
from model import Classifier
import torch

data_path = "/home/oliver/random/Download Data - STOCK_US_XNAS_AAPL.csv"

data = MyDataset(data_path)
model = Classifier()
trainer = Trainer(model=model,
                data=data,
                )
trainer.train(100)
trainer.save_model()