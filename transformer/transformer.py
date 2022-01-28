from turtle import forward
from typing import ForwardRef, Pattern
from torch import Tensor, nn, softmax
import torch.nn.functional as F
from torch.nn import Transformer
import torch
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, nhead=8, dmodel=512, dk=64, dv=64):
        super().__init__()
        self.nhead = nhead
        self.WQ = []
        self.WK = []
        self.WV = []
        for i in range(nhead):
            self.WQ.append(torch.nn.Parameter(torch.rand((dmodel,dk),requires_grad=True)))
            self.WK.append(torch.nn.Parameter(torch.rand((dmodel,dk),requires_grad=True)))
            self.WV.append(torch.nn.Parameter(torch.rand((dmodel,dv),requires_grad=True)))
        self.WO = torch.nn.Parameter(torch.rand((nhead*dv, dmodel),requires_grad=True))
    
    def forward(self, Q:Tensor, K:Tensor, V:Tensor):
        #  这个输出是 seq_len* dmodel
        ohead = []
        for i in range(self.nhead):
            Qi = Q @ self.WQ[i]
            Ki = K @ self.WK[i]
            Vi = V @ self.WV[i]
            dk = Ki.size()[-1]
            headi = F.softmax(Qi @ Ki.transpose(1,2) / math.sqrt(dk)) @ Vi # FIXME:这里有警告
            ohead.append(headi)
        output = torch.cat(ohead, -1)
        output = output @ self.WO
        return output
        
class transformerv1(nn.Module):
    def __init__(self, seq_len=128, nfeature=512):
        super().__init__()
        self.seq_len = seq_len
        self.dmodel = 512
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
        self.pool = nn.MaxPool2d(16, stride = 16)
        self.FC = nn.Linear(256, 2) #FIXME:这里的参数针对默认self.dmodel = 512

    def forward(self, input):
        input = self.emb(input)
        input = input.view(-1, self.seq_len, self.dmodel)
        input = input + self.posEncoder(input.size()[0],input.size()[1],input.size()[2])
        x = self.multiHeadAttention(input, input, input)
        x = self.dropout(x)+input
        x = self.norm(x)
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = self.pool(x)
        x = x.reshape(-1,256)#  FIXME: 这里默认维度才对哦self.dmodel = 512
        x = self.FC(x)
        x = F.log_softmax(x,dim=-1)
        return x

    def posEncoder(self, nbatch:int, npos:int, nmodel:int):
        PE = []
        for pos in range(npos):
            PEpos = []
            for i in range(nmodel/2):
                PEpos.append(math.sin(float(pos)/pow(10000.0, 2.0*i/float(nmodel))))
                PEpos.append(math.cos(float(pos)/pow(10000.0, 2.0*i/float(nmodel))))
            PE.append(PEpos)
        return Tensor([PE*nbatch])

class transformerv2(nn.Module):
    def __init__(self, seq_len=128, nfeature=512):
        super().__init__()
        self.seq_len = seq_len
        self.dmodel = 512
        # embedding
        self.emb = nn.Linear(nfeature,512)
        # encoder
        self.transformerEcoder = nn.TransformerEncoderLayer(self.dmodel,8)
        # classify
        self.pool = nn.MaxPool2d(16, stride = 16)
        self.FC = nn.Linear(256, 2) #FIXME:这里的参数针对默认self.dmodel = 512

    def forward(self,input):
        input = self.emb(input)
        input = input.view(-1, self.seq_len, self.dmodel)
        x = self.transformerEcoder(input)
        x = self.pool(x)
        x = x.reshape(-1,256)#  FIXME: 这里默认维度才对哦self.dmodel = 512
        x = self.FC(x)
        x = F.log_softmax(x,dim=-1)
        return x