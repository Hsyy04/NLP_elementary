from numpy import save
from bert.bert import bert
from bert.make_data import BertDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import torch.nn.functional as F
import time
from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter
# from torchstat import stat
from torchsummary import summary
import os
from transformers import WEIGHTS_NAME, CONFIG_NAME

os.environ['CUDA_VISIBLE_DEVICES']='1'  # here chose the GPU
torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
device = torch.device(device)

def cal_mask_loss(pre_output:torch.Tensor, token: torch.Tensor, mask_token:torch.Tensor, weights:torch.Tensor):
    """caculate ML loss

    Args:
        pre_output (tensor): the whole output of the Network([batch_size, seq_length, H_size])
        mask_token (tensor): the ids of the masked word [batch_size, len(mask_token)]
        weights (tensor): embedding weights [voca_size, H_size]
        token(tensor): the no-mask id [batch_size, seq_length]
    Returns:
        loss(double) 
    """  
    _, batch_size, H_size= pre_output.shape
    bf_preoutput = pre_output.transpose(0,1)
    mask_len = mask_token.shape[1]
    voca_size = weights.shape[0]

    # the predict words
    pre_mask = torch.gather(bf_preoutput, 1, mask_token.unsqueeze(-1).tile((1,1,H_size)))  # [batch_size, len(mask_token), H_size]
    pred_probs = torch.inner(pre_mask, weights) # [batch_size, len(mask_token), voca_size]
    pred_probs = F.log_softmax(pred_probs, -1)
   
    # loss
    loss_vec =torch.gather(pred_probs, -1, mask_token.unsqueeze(-1)).squeeze(-1) #[batch_size, len(mask_token)]
    loss_vec*=(-1.0)
    ok_matrix = torch.zeros(size=(batch_size, mask_len),device=device)
    loss_vec = torch.where(mask_token>0, loss_vec, ok_matrix) # 这里过滤掉补齐长度的部分
    ok_matrix = torch.where(mask_token>0, ok_matrix+1, ok_matrix).sum(-1).unsqueeze(-1)
    loss = (loss_vec/ok_matrix).sum(-1).sum(-1)
    return loss

def cal_next_loss(pre_output:torch.Tensor, target):
    pred = F.log_softmax(pre_output, -1)
    loss = F.nll_loss(pred, target)
    return loss

BATCH_SIZE = 16
EPOCH_NUM = 100
LEARNING_RATE = 0.0003
PADDING_LEN = 256   # i.e.seq_len
NAME = 'bert_pretrain'
output_dir = "./model/bert/"


data_set = BertDataset(max_len=PADDING_LEN)
dic_size = data_set.tokenizer.vocab_size
train_data = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True)
print(f"load data successful")

model = bert(dic_size, PADDING_LEN).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

start = time.time()
save_loss=1000000

for epoch in range(EPOCH_NUM):
    model.train()
    totloss = 0.0
    print(f"Epoch: {epoch+1}/{EPOCH_NUM}")
    for X, all_id, Y1, Y2 in tqdm(train_data, mininterval=30): 
        X, Y1, all_id, Y2= X.to(device), Y1.to(device), all_id.to(device), Y2.to(device)
        output, next_sent_output= model(X[:,0], X[:,1], X[:,2], X[:,3])
        loss = cal_mask_loss(output, all_id, Y1, model.emb_token.weight)
        loss += cal_next_loss(next_sent_output, Y2)
        totloss+=loss
        optimizer.zero_grad() # 这个是梯度置零 添加set_to_none可以置为None，会占用更小的内存，但是会出事
        loss.backward()
        optimizer.step()
    print(f"loss:{totloss}\n")
    if save_loss >= totloss:
        save_loss = totloss
        torch.save(model.state_dict(), f'model/bert/{NAME}.pt')
        torch.save(model, f'model/bert/{NAME}_all.pt')

end = time.time()
print(f"time:{end-start}s")
print(f"result:{save_loss}")
