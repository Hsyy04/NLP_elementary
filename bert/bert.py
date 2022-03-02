from turtle import forward
from transformers import BertModel,BertConfig
from torch import Tensor, nn, tensor
import torch.nn.functional as F

class bertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # model_config = BertConfig.from_pretrained('BERT-wwm', output_hidden_states=True)
        self.bert = BertModel.from_pretrained('hfl/chinese-bert-wwm')
        # FIXME:这都是根据bert\chinese_bert_wwm_L-12_H-768_A-12\publish\\bert_config.json调的
        self.dropout = nn.Dropout(0.1)
        self.pool = nn.MaxPool1d(16, stride = 16)
        self.FC = nn.Linear(768//16,2)

    def forward(self,input:Tensor):
        # FIXME:这个bert出来是什么？
        x:Tensor = self.bert(input)
        x = x['last_hidden_state'][:,0,:]

        x = self.dropout(x)
        x = x.unsqueeze(1)
        x = self.pool(x)
        x = self.FC(x)
        x = F.log_softmax(x,dim=-1).squeeze(1)
        return x
