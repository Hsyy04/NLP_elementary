from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import StepLR

from data import bertEmbedding, ChSentiDataSet
from bert.bert import bert, bertClassifierv2
from torch.utils.data import DataLoader,random_split

import torch
from tqdm import tqdm
import torch.nn.functional as F
import time
import torchvision.models as models
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
# from torchstat import stat
# from torchsummary import summary
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'
torch.cuda.empty_cache()
TRAIN_PATH = "data/ChnSentiCorp_htl_all/train_1600+1600.csv"
TEST_PATH = "data/ChnSentiCorp_htl_all/test_800+800.csv"
BATCH_SIZE = 8
EPOCH_NUM = 40
LEARNING_RATE = 0.00003
PADDING_LEN = 256   # i.e.seq_len
DROUP_OUT = 0.7
NAME = f'ft_bert_lr{LEARNING_RATE}_en{EPOCH_NUM}_adam_d{DROUP_OUT}_pl{PADDING_LEN}'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
device = torch.device(device)

class myOptimSimple(Optimizer):
    
    def __init__(self, params, lr) -> None:
        default = dict(lr=lr)
        super().__init__(params, default)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is not None:
                    p.data.add_(p.grad,alpha=-lr)
        return 0

def valid(data, model, phrase):
    model.eval()
    currect = 0
    all = 0
    loss = 0.0
    pos_id = torch.arange(PADDING_LEN).to(device)
    for (X,y) in data:
        all+=1
        X = X.to(device).unsqueeze(0)
        y_pred = model(X[:,0,:], pos_id, X[:,1,:], X[:,2,:])
        y_pred = y_pred.squeeze(0)
        loss += (-float(y_pred[y]))
        y_pred = torch.argmax(y_pred,dim=-1)
        if y_pred == y:
            currect+=1
    print(phrase)
    acc = float(currect)*100.0/float(all)
    print(f"acc: {currect}/{all}:({acc}%)")
    return acc, loss/float(len(data))

if __name__ == "__main__":
    # const valuables
    currentTime = datetime.now().strftime('%b%d_%H-%M-%S')

    # ????????????
    eb = bertEmbedding("data/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv",PADDING_LEN)
    dicSize = len(eb)
    print(f"dictionary size:{len(eb)}")
    train_data = ChSentiDataSet(TRAIN_PATH,eb)
    test_data = ChSentiDataSet(TEST_PATH,eb)
    # train data
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    start = time.time()
    # ????????????
    model = bertClassifierv2(dicSize, PADDING_LEN)
    model.load_state_dict(torch.load('model/bert/bert_pretrain.pt'), strict=False)
    #FIXME: stat(model,(1,128,727))??????????????????
    # summary(model.cuda(),input_size=(1,128,727),batch_size=64)
    # assert(False)

    # ?????????
    # optimizer = myOptimSimple(model.parameters(), lr=LEARNING_RATE)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=153, gamma=0.5)

    writer = SummaryWriter(f"runs/{NAME}_{currentTime}/")
    best = 0
    pos_id = torch.arange(PADDING_LEN).to(device)
    model.to(device)
    for epoch in range(EPOCH_NUM):
        model.train()
        totloss = 0.0
        print(f"Epoch: {epoch+1}/{EPOCH_NUM}")
        for (X, y_std) in tqdm(train_dataloader):
        # for (X, y_std) in train_dataloader:
            X, y_std = X.to(device), y_std.to(device)
            y_pred = model(X[:,0,:], pos_id, X[:,1,:], X[:,2,:]) # if bert!
            loss = F.nll_loss(y_pred, y_std) 
            totloss+=loss
            optimizer.zero_grad() # ????????????????????? ??????set_to_none????????????None?????????????????????????????????????????????
            loss.backward()
            optimizer.step()

        print(f"loss:{totloss}")
        writer.add_scalars('train/loss', {'loss':totloss}, epoch)
        if totloss < 70.0:
            val_acc,val_loss = valid(train_data, model,"vaild")
            test_acc,test_loss = valid(test_data, model,"test")
            dic = {'val':val_loss,'test':test_loss}
            writer.add_scalars('result/loss', dic, epoch)
            dic = {'val':val_acc,'test':test_acc}
            writer.add_scalars('result/acc', dic, epoch)

            if best < test_acc:
                best = test_acc
                torch.save(model, f"model/{NAME}.pth") 
                print("saving...")
        print('\n')

        scheduler.step()

    if best < 0.5:
        val_acc,val_loss = valid(train_data, model,"vaild")
        test_acc,test_loss = valid(test_data, model,"test")
    else:
        model = torch.load(f"model/{NAME}.pth")
        val_acc,val_loss = valid(train_data, model,"vaild")
        test_acc,test_loss = valid(test_data, model,"test")

    end = time.time()
    print(f"time:{end-start}\n")
     
    writer.close()