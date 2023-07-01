import torch.nn as nn
import torch
from utils.snake import Snake




def loss(pred, target):
    squares = (pred - target)**2
    return squares.mean()


class Net_Line_regres(torch.nn.Module):
    def __init__(self, input_feature): #  
        super().__init__()
        self.fc1 = nn.Linear(input_feature, 1) 
    def forward(self, x):
        x = self.fc1(x)
        return x


class Net_Snake(torch.nn.Module):
    def __init__(self, input_feature, alpha): 
        super().__init__()
        self.fc1 = nn.Linear(input_feature, 1) 
        self.act_snake = Snake(1, alpha)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act_snake(x)
        return x


