import torch
import torch.nn as nn
from torch import Tensor
class Classifier(nn.Module):
    """
    A simple class that represents the classifier model (logistic regression).
    """
    
    def __init__(self,
                input_dim: int = 3, 
                output_dim: int = 1
                ):
        super().__init__()
        self.linear = nn.Linear(input_dim,output_dim)
        self.sigmoid = nn.Sigmoid()
     
    def forward(self,
                x: Tensor
                ):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x