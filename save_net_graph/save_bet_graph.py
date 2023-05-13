from torchview import draw_graph

import torch.nn as nn
import torch

class Net(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super().__init__()
        # YOUR CODE HERE
        self.fc1 = nn.Linear(1, n_hidden_neurons)
        self.act_relu = nn.ReLU() 
        self.fc2 = nn.Linear(n_hidden_neurons, 1)
        pass

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_relu(x)
        x = self.fc2(x)
        return x


def loss(pred, target):
    squares = (pred - target)**2
    return squares.mean()

from torchview import draw_graph

model = Net(512)
batch_size = 1
# device='meta' -> no memory is consumed for visualization
model_graph = draw_graph(model, input_size=(batch_size, 1), device='meta',    save_graph=True, filename="here")
model_graph.visual_graph