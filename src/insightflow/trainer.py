import torch
from torch.utils.data import DataLoader
import torch.nn as nn 
from torch import optim 
from insightflow.data_processing import MyDataset
from insightflow.model import Classifier
from pathlib import Path

class Trainer():
    def __init__(self, 
                 model, 
                 data, 
                 val_split_size: float = 0.0, 
                 learning_rate: float = 0.001, 
                 batch_size: int = 1,
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.train_data = DataLoader(data, batch_size=batch_size, shuffle=True)
        self.val_data = None
        self.val_split_size = val_split_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()
        self.path: str | None = None

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        
        for inputs, labels in self.train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            y_hat = self.model(inputs)
            loss = self.criterion(y_hat.squeeze(1), labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        
        return running_loss / len(self.train_data)

    def evaluate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                y_hat = self.model(inputs)
                loss = self.criterion(y_hat.squeeze(1), labels)
                running_loss += loss.item()
                predicted = torch.round(torch.sigmoid(y_hat))
                correct += (predicted.squeeze() == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = running_loss / len(self.val_data)
        accuracy = correct / total
        
        return avg_loss, accuracy

    def train(self, 
            num_epochs: int = 10
            ):
        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            train_loss = self.train_one_epoch()
            print(f'Train loss: {train_loss:.4f}')
            
            # eval_loss, eval_acc = self.evaluate()
            # print(f'Eval loss: {eval_loss:.4f}; Eval Accuracy: {eval_acc:.4f}')
    
    def save_model(self, path='model_weights.path'):
        try:
            parent_dict = Path(__file__).parent
            full_path = parent_dict / path
            torch.save(self.model.state_dict(), full_path)
            print(f'model weights saved at {full_path}')
            self.path = full_path
        except Exception as e:
            print(f'Error saving model: {e}')
