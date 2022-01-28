from typing import ForwardRef, Pattern
from torch import nn
import torch.nn.functional as F
import torch

class textCNNv1(nn.Module):
    def __init__(self, input_size=(128,727), dropout=0.5) -> None:
        super().__init__()
        self.cnn3 = nn.Conv2d(in_channels=1,out_channels=100,kernel_size=(3,input_size[1]))
        self.pool3 = nn.MaxPool1d(kernel_size=input_size[0]-3+1, stride=1)
        self.cnn4 = nn.Conv2d(in_channels=1,out_channels=100, kernel_size=(4,input_size[1]))
        self.pool4 = nn.MaxPool1d(kernel_size=input_size[0]-4+1, stride=1)
        self.cnn5 = nn.Conv2d(in_channels=1,out_channels=100,kernel_size=(5,input_size[1]))
        self.pool5 = nn.MaxPool1d(kernel_size=input_size[0]-5+1, stride=1)
        self.droupout = nn.Dropout(p=dropout)
        self.FC = nn.Linear(300,2)

    def forward(self, input):
        x3 = F.relu(self.cnn3(input))
        x3 = x3.squeeze(-1)
        x3 = self.pool3(x3)
        x4 = self.pool4(F.relu(self.cnn4(input)).squeeze(-1))
        x5 = self.pool5(F.relu(self.cnn5(input)).squeeze(-1))
        x = torch.cat((x3,x4,x5),dim=1).squeeze(-1)
        x = self.droupout(x)
        x = self.FC(x)
        x = F.log_softmax(x,dim=-1)
        return x

        
