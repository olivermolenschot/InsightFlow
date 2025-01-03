import torch
from insightflow.data_processing import MyDataset
from insightflow.trainer import Trainer
from insightflow.model import Classifier

#data_path = "/home/oliver/random/Download Data - STOCK_US_XNAS_AAPL.csv"

def dataset_to_model(batch_size: int,
                    data_path: str,
                    num_epochs: int = 10,
                    learning_rate: float = 0.001,
                    device: str = 'cpu',
                    )-> Trainer:
    data = MyDataset(data_path)
    model = Classifier()
    trainer = Trainer(model=model,
                    data=data,
                    learning_rate= learning_rate,
                    device= device,
                    batch_size=batch_size
                    )
    trainer.train(num_epochs = num_epochs)
    trainer.save_model()
    return trainer