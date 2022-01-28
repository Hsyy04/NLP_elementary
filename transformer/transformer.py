from typing import ForwardRef, Pattern
from torch import Tensor, nn, softmax
import torch.nn.functional as F
import torch
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, nhead=8, dmodel=512, dk=64, dv=64):
        super().__init__()
        self.nhead = 8
        self.WQ = []
        self.WK = []
        self.WV = []
        for i in range(nhead):
            self.WQ.append(torch.nn.Parameter(torch.rand((dmodel,dk))))
            self.WK.append(torch.nn.Parameter(torch.rand((dmodel,dk))))
            self.WV.append(torch.nn.Parameter(torch.rand((dmodel,dv))))
        self.WO = torch.nn.Parameter(torch.rand((nhead*dv, dmodel)))
    
    def forward(self, Q:Tensor, K:Tensor, V:Tensor):
        #  这个输出是 seq_len* dmodel
        ohead = []
        for i in range(self.nhead):
            Qi = Q @ self.WQ[i]
            Ki = K @ self.WK[i]
            Vi = V @ self.WV[i]
            dk = Ki.shape()[1]
            headi = F.softmax(Qi @ Ki.transpose(0,1) / math.sqrt(dk)) @ Vi
            ohead.append(headi)
        output = torch.cat(ohead, 1)
        output = output @ self.WO
        return output
        
class transformerv1(nn.Module):
    def __init__(self, seq_len=128, nfeature=512):
        super().__init__()
        # embedding
        self.emb = nn.Linear(nfeature,512)
        dmodel = 512
        # encoder
        self.multiHeadAttention = MultiHeadAttention(dmodel = dmodel)
        self.dropout = nn.Dropout(p=0.1)
        self.norm = nn.LayerNorm((seq_len,dmodel))
        self.l1 = nn.Linear(dmodel, dmodel*4)
        self.l2 = nn.Linear(dmodel*4, dmodel)
        # 分类
        self.pool = nn.MaxPool1d(16, stride = 16)
        self.FC = nn.Linear(256, 2) #FIXME:这里的参数针对默认

    def forward(self, input):
        input = self.emb(input)
        x = self.multiHeadAttention(input, input, input)
        x = self.dropout(x)+input
        x = self.norm(x)
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = self.pool(x)
        x = self.FC(x)
        x = F.log_softmax(x,dim=-1)
        return x



    