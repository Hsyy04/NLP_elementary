from turtle import forward, position
from typing import ForwardRef, Pattern
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn import Transformer, MultiheadAttention
import torch
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 515):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, nhead=8, dmodel=512, p=0.1):
        super().__init__()
        dk = dmodel//nhead
        dv = dmodel//nhead
        self.nhead = nhead
        self.WQ = []
        self.WK = []
        self.WV = []
        for i in range(nhead):
            self.WQ.append(torch.nn.Parameter(torch.rand((dmodel,dk),requires_grad=True, device=torch.device('cuda:0'))))
            self.WK.append(torch.nn.Parameter(torch.rand((dmodel,dk),requires_grad=True, device=torch.device('cuda:0'))))
            self.WV.append(torch.nn.Parameter(torch.rand((dmodel,dv),requires_grad=True, device=torch.device('cuda:0'))))
        self.WO = torch.nn.Parameter(torch.rand((nhead*dv, dmodel),requires_grad=True, device=torch.device('cuda:0')))
        self.p = p
        self.dropout = nn.Dropout(p)

    def forward(self, Q:Tensor, K:Tensor, V:Tensor):
        #  这个输出是 seq_len* dmodel
        ohead = []
        for i in range(self.nhead):
            Qi = Q @ self.WQ[i]
            Ki = K @ self.WK[i]
            Vi = V @ self.WV[i]
            dk = Ki.size()[-1]
            headi = F.softmax(torch.bmm(Qi , Ki.transpose(1,2)) / math.sqrt(dk),dim=-1)
            headi = F.dropout(headi,self.p)
            headi = torch.bmm(headi , Vi)
            ohead.append(headi)
        output = torch.cat(ohead, -1)
        output = output @ self.WO
        return output

class posEncoder(nn.Module):
    def __init__(self, npos:int, nmodel:int,dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(npos, nmodel)
        position = torch.arange(0,npos).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, nmodel, 2).float()*(-math.log(10000.0)/float(nmodel)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # 这种方式也可以把参数放到gpu？

    def forward(self, x):
        return self.dropout(x+self.pe)

class transformerv1(nn.Module):
    def __init__(self, seq_len=128, nfeature=512, p=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.dmodel = 512
        # embedding
        self.emb = nn.Linear(nfeature,self.dmodel)
        # 生成pe
        self.pe = posEncoder(seq_len,self.dmodel)
        # encoder
        self.multiHeadAttention = MultiHeadAttention(dmodel = self.dmodel, nhead=4)
        # self.multiHeadAttention = MultiheadAttention(self.dmodel, 4, dropout=p, batch_first=True)
        self.dropout1 = nn.Dropout(p)
        self.norm1 = nn.LayerNorm(self.dmodel)

        self.l1 = nn.Linear(self.dmodel, self.dmodel*4)
        self.dropout = nn.Dropout(p)
        self.l2 = nn.Linear(self.dmodel*4, self.dmodel)
        self.dropout2 = nn.Dropout(p)
        self.norm2 = nn.LayerNorm(self.dmodel)
        # classify
        self.pool = nn.MaxPool1d(16, stride = 16)
        self.FC = nn.Linear(self.seq_len*int(self.dmodel/16),2)

        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.FC.weight.data.uniform_(-initrange, initrange)
        self.FC.bias.data.zero_()

    def forward(self, input):
        input = self.emb(input)
        input = input.view(-1, self.seq_len, self.dmodel)
        input = self.pe(input)
        x = self.multiHeadAttention(input, input, input)
        x = self.dropout1(x)+input
        x = self.norm1(x)

        x2 = self.l2(self.dropout(F.relu(self.l1(x))))
        x = self.dropout2(x2)+x
        x = self.norm2(x)

        x = self.pool(x)
        x = x.reshape(-1,self.seq_len*int(self.dmodel/16))
        x = self.FC(x)
        x = F.log_softmax(x,dim=-1)
        return x

class transformerv2(nn.Module):
    def __init__(self, dic_size, seq_len=128, nfeature=512):
        super().__init__()
        self.seq_len = seq_len
        self.dmodel = 512
        # embedding使用torch的！！！！！！！
        self.emb = nn.Embedding(dic_size, self.dmodel)
        # pos
        self.pe = posEncoder(seq_len, nmodel=self.dmodel)
        # encoder
        layer =  nn.TransformerEncoderLayer(self.dmodel,nhead=4,batch_first=True)
        self.transformerEcoder = nn.TransformerEncoder(layer,1)
        # classify
        self.pool = nn.MaxPool1d(16, stride = 16)
        self.FC = nn.Linear(self.seq_len*int(self.dmodel/16),2)
    
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.FC.weight.data.uniform_(-initrange, initrange)
        self.FC.bias.data.zero_()

    def forward(self,input):
        input = self.emb(input)
        input = input.reshape(-1, self.seq_len, self.dmodel)
        input = self.pe(input)
        x = self.transformerEcoder(input)
        x = self.pool(x)
        x = x.reshape(-1,self.seq_len*int(self.dmodel/16))
        x = self.FC(x)
        x = F.log_softmax(x,dim=-1)
        return x