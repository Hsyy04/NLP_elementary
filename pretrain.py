from calendar import calendar
import imp
from sklearn import preprocessing
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import StepLR
from bert.bert import bert
from bert.make_data import BertDataset
from torch.utils.data import DataLoader,random_split
import torch
from tqdm import tqdm
import torch.nn.functional as F
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
# from torchstat import stat
# from torchsummary import summary
import os

def cal_mask_loss(pre_output:torch.Tensor, token: torch.Tensor, mask_token:torch.Tensor, weights:torch.Tensor):
    """caculate ML loss

    Args:
        pre_output (tensor): the whole output of the Network([batch_size, seq_length, H_size])
        mask_token (tensor): the ids of the masked word [batch_size, len(mask_token)]
        weights (tensor): embedding weights [voca_size, H_size]

    Returns:
        loss(double) 
    """    
    batch_size, _, H_size= pre_output.shape
    mask_len = mask_token.shape[1]
    # voca_size = weights.shape[0]
    # the predict words
    pre_mask =torch.Tensor(size=(batch_size, mask_len, H_size)).gather(1, mask_token, pre_output)  # [batch_size, len(mask_token), H_size]
    pred_probs = torch.inner(pre_mask, weights) # [batch_size, len(mask_token), vac_size]
    pred_probs = F.log_softmax(pred_probs, -1)
    # the true words
    word_ids = torch.Tensor(size=(batch_size, mask_len)).gather(-1, mask_token, pre_output)
    # loss
    loss = F.cross_entropy(pred_probs,word_ids)
    return loss

def cal_next_loss(pre_output:torch.Tensor, target):
    pred = F.log_softmax(pre_output[:,:,0])
    loss = F.cross_entropy(pred, target)
    return loss

BATCH_SIZE = 8
EPOCH_NUM = 40
LEARNING_RATE = 0.00003
PADDING_LEN = 512   # i.e.seq_len

os.environ['CUDA_VISIBLE_DEVICES']='0'  # here chose the GPU
torch.cuda.empty_cache()

data_set = BertDataset()
dic_size = len(data_set)
train_data = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True)

model = bert(dic_size, PADDING_LEN)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

start = time.time()

for epoch in range(EPOCH_NUM):
    model.train()
    totloss = 0.0
    print(f"Epoch: {epoch+1}/{EPOCH_NUM}")
    for X, all_id, Y1, Y2 in tqdm(train_data):
        output = model(X)
        loss = cal_mask_loss(output, all_id, Y1, model.emb_token.parameters())
        loss += cal_next_loss(output, Y2)
        totloss+=loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"loss:{totloss}\n")

end = time.time()
elapsed = str(datetime.timedelta(seconds=end-start))
print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))
