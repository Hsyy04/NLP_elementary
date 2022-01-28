from torch import nn
import torch.nn.functional as F
import torch

# 使用全连接层实现
class MLPmodelV1(nn.Module):
    def __init__(self, inputsize=350) -> None:
        super().__init__()
        self.hidden1 = nn.Linear(inputsize, 128, bias=True)
        self.hidden2 = nn.Linear(128, 64, bias=True)
        self.output = nn.Linear(64, 2)

    def forward(self, input):
        x = F.relu(self.hidden1(input.float()))
        x = F.relu(self.hidden2(x))
        x = F.log_softmax(self.output(x),dim=-1)
        return x

class MLPmodelV2(nn.Module):
    def __init__(self, inputsize=350) -> None:
        super().__init__()
        self.w1 =  torch.nn.Parameter(torch.rand((inputsize,64),requires_grad=True,dtype=torch.float64))
        self.b1 = torch.nn.Parameter(torch.rand((64,),requires_grad=True,dtype=torch.float64))
        self.wo = torch.nn.Parameter(torch.rand((64,2),requires_grad=True,dtype=torch.float64))

    def forward(self, input):
        x = input @ self.w1 + self.b1
        x = F.relu(x)
        x = x @ self.wo
        x = F.log_softmax(x,dim=-1)
        return x