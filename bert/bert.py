from transformers import BertModel,BertConfig
from torch import Tensor, nn, tensor
import torch.nn.functional as F

class bertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('hfl/chinese-bert-wwm')
        # FIXME:这都是根据hfl/chinese-bert-wwm写的参数
        self.dropout = nn.Dropout(0.1)
        self.pool = nn.MaxPool1d(16, stride = 16)
        self.FC = nn.Linear(768//16,2)

    def forward(self,input:Tensor):
        x:Tensor = self.bert(input[:,0,:], **{'token_type_ids':input[:,1,:],'attention_mask':input[:,2,:]})
        x = x['last_hidden_state'][:,0,:]
        x = self.dropout(x)
        x = x.unsqueeze(1)
        x = self.pool(x)
        x = self.FC(x)
        x = F.log_softmax(x,dim=-1).squeeze(1)
        return x

class bert(nn.Module):
    def __init__(self, dict_size, seq_length, H_size=768, L_size=12, A_size=12):
        super().__init__()
        self.emb_token = nn.Embedding(dict_size, H_size)
        self.emb_pos = nn.Embedding(seq_length, H_size)
        self.emb_segment = nn.Embedding(2, H_size)
        layer =  nn.TransformerEncoderLayer(H_size,nhead=A_size,batch_first=True)
        self.transformerEcoder = nn.TransformerEncoder(layer,L_size) 

    def forward(self, token_input, pos_input, seg_input, mask):
        token_x = self.emb_token(token_input)
        pos_x = self.emb_pos(pos_input)
        seg_x = self.emb_segment(seg_input)
        x = token_x + pos_x + seg_x
        x = self.transformerEcoder(x,attention_mask=mask)
        return x

if __name__ == "__main__":

    pass