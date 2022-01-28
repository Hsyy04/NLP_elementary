from sys import float_repr_style
from unicodedata import bidirectional
from torch import nn
import torch.nn.functional as F
import torch

class LSTMv1(nn.Module):
    def __init__(self, input_features, hidden_size, seq_length, hidden_layers=1, dropout=0):
        super().__init__()
        
        self.input_size = (seq_length,input_features)
        self.output_size = hidden_size*seq_length*2
        self.lstm = nn.LSTM(input_size=input_features, hidden_size=hidden_size, batch_first=True,  num_layers=hidden_layers, dropout=dropout, bidirectional=True)
        self.pool = nn.MaxPool1d(16,stride=16)
        self.FC = nn.Linear(512,2)

    def forward(self, input):
        input = input.view(-1,self.input_size[0], self.input_size[1])
        x, _= self.lstm(input)
        x = x.reshape(-1, 1, self.output_size)
        x = self.pool(x)
        x = x.reshape(-1,512)
        x = self.FC(x)
        x = F.log_softmax(x,dim=-1)
        return x

class GRUv1(nn.Module):
    def __init__(self, input_features, hidden_size, seq_length, hidden_layers=1, dropout=0):
        super().__init__()
        self.emb = nn.Linear(input_features, 512)
        input_features = 512
        self.input_size = (seq_length,input_features)
        self.output_size = hidden_size*seq_length*2
        self.gru = nn.GRU(input_size=input_features, hidden_size=hidden_size, batch_first=True,  num_layers=hidden_layers, dropout=dropout, bidirectional=True)
        self.pool = nn.MaxPool1d(16,stride=16)
        self.FC = nn.Linear(512,2)

    def forward(self, input):
        input = self.emb(input)
        input = input.view(-1,self.input_size[0], self.input_size[1])
        x, _= self.gru(input)
        x = x.reshape(-1, 1, self.output_size)
        x = self.pool(x)
        x = x.reshape(-1,512)
        x = self.FC(x)
        x = F.log_softmax(x,dim=-1)
        return x 