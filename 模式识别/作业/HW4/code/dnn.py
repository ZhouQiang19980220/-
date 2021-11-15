import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, out_channels)
        
    def forward(self,x):
        x = torch.tanh(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        
        return x