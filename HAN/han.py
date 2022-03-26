from cmath import tanh
from turtle import forward
from unicodedata import bidirectional
from torch import Tensor, dropout, nn
import torch.nn.functional as F
import torch

class han(nn.Module):
    def __init__(self, dict_size, doc_len, sent_len, hidden_size=128):
        super().__init__()
        # word encoder
        self.word_emb = nn.Embedding(dict_size, hidden_size)
        self.word_GRU = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True, dropout=0.5, num_layers=2)
        # word attention 
        self.word_attention = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        # sentence encoder
        self.sent_GRU = nn.GRU(hidden_size*2, hidden_size, bidirectional=True, batch_first=True, dropout=0.5, num_layers=2)
        # sentence attention
        self.sent_attention = nn.Sequential(
            # [batch_size, doc_len, sent_len, hidden_size*2]
            nn.Linear(hidden_size*2, hidden_size),
            nn.Tanh(),
            # [batch_size, doc_len, sent_len, hidden_size]
            nn.Linear(hidden_size, 1)
            # [batch_size, doc_len, sent_len, 1]
        )
        # classifier
        self.cls = nn.Linear(hidden_size*2, 2)
        self.doc_len = doc_len
        self.sent_len = sent_len
        self.hidden_size = hidden_size

    def forward(self, input:Tensor, vis = False):
        # input : [batch_size, doc_len, sent_len]
        x = self.word_emb(input).reshape(-1, self.doc_len* self.sent_len, self.hidden_size) # [batch_size, doc_len, sent_len, hidden_size]
        x, _= self.word_GRU(x)
        x = x.reshape(-1, self.doc_len, self.sent_len, self.hidden_size*2) # x=[batch_size, doc_len, sent_len, hidden_size*2]
        a = self.word_attention(x)
        a = F.softmax(a, dim=2) # [batch_size, doc_len, sent_len]
        x = a * x 
        x = x.sum(dim=2) #  [batch_size, doc_len, hidden_size*2]
        x, _ = self.sent_GRU(x) #  [batch_size, doc_len, hidden_size*2]
        a = self.sent_attention(x)
        a=F.softmax(a, dim=1)
        x = a * x 
        x = x.sum(dim=1) 
        x = F.log_softmax(self.cls(x),dim=-1)
        
        return x





