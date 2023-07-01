import torch.nn as nn
import torch
from utils.snake import Snake


# class One_layer(torch.nn.Module):
#     def __init__(self, input_feature):
#         super().__init__()
#         # YOUR CODE HERE
#         # self.fc1 = nn.Linear(input_feature, 1)
#         self.act_snake = Snake(input_feature, 4)
#         self.fc1 = nn.Linear(input_feature, 1)
#         # # self.bn=nn.BatchNorm1d(input_feature)
#         # self.fc2 = nn.Linear(input_feature, 1)
#         self.act_relu = nn.ReLU()
#         # self.fc3 = nn.Linear(16, 1)
#         pass

#     def forward(self, x):
#         x =self.act_snake(x)
#         x = self.fc1(x)
#         # x = self.act_snake(x)
#         # # # # x = self.bn(x)
#         # x = self.fc2(x)
#         x = self.act_relu(x)
#         # x = self.fc3(x)
#         return x


def loss(pred, target):
    squares = (pred - target)**2
    return squares.mean()



class Net(torch.nn.Module):
    def __init__(self, input_feature,n_hidden_neurons): # , n_hidden_neurons, alpha
        super().__init__()
        # YOUR CODE HERE
        self.fc1 = nn.Linear(input_feature, 1) # n_hidden_neurons
        # self.act_relu = nn.ReLU() 
        # # self.act_snake = Snake(n_hidden_neurons, alpha)
        # self.fc2 = nn.Linear(1, 42)
        # self.act_relu = nn.ReLU() 
        # self.fc3 = nn.Linear(42, 1)
        self.sig = nn.Sigmoid()
        pass

    def forward(self, x):
        x = self.fc1(x)
        # x = self.act_relu(x)
        # x = self.fc2(x)
        # x = self.act_relu(x)
        # x = self.fc3(x)
        # x = self.act_snake(x)
        # x = self.fc2(x)
        # x = self.sig(x)
        x = self.sig(x)

        return x


# def loss(pred, target):
#     squares = (pred - target)**2
#     return squares.mean()



# class SnakeNet(torch.nn.Module):
#     def __init__(self,input_feature, n_hidden_neurons,n_hidden_neurons_snake,alpha):
#         super().__init__()
#         # YOUR CODE HERE
#         self.conv1 = nn.Conv1d(42, 42, 7, stride=2)

#         self.fc1 = nn.Linear(input_feature, n_hidden_neurons)
#         self.act_relu = nn.ReLU() 
#         self.fc2 = nn.Linear(n_hidden_neurons, n_hidden_neurons_snake)
#         self.act_snake = Snake(n_hidden_neurons_snake, alpha)
#         self.fc3 = nn.Linear(n_hidden_neurons_snake, 1)
#         pass

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act_relu(x)
#         x = self.fc2(x)
#         x = self.act_snake(x)
#         x = self.fc3(x)
#         return x




# class SnakeNet(torch.nn.Module):
#     def __init__(self,input_feature, n_hidden_neurons,n_hidden_neurons_snake,alpha):
#         super().__init__()
#         # YOUR CODE HERE
#         self.fc1 = nn.Linear(input_feature, n_hidden_neurons)
#         self.act_relu = nn.ReLU() 
#         self.fc2 = nn.Linear(n_hidden_neurons, n_hidden_neurons_snake)
#         self.act_snake = Snake(n_hidden_neurons_snake, alpha)
#         self.fc3 = nn.Linear(n_hidden_neurons_snake, 1)
#         pass

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act_relu(x)
#         x = self.fc2(x)
#         x = self.act_snake(x)
#         x = self.fc3(x)
#         return x


