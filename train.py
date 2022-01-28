from torch.optim.optimizer import Optimizer
from data import embedding, ChSentiDataSet, oneHotEmbedding
from MLP.MLP import MLPmodelV1, MLPmodelV2
from TextCNN.textCNN import textCNNv1
from RNN.rnn import LSTMv1, GRUv1
from transformer.transformer import transformerv1
from torch.utils.data import DataLoader,random_split
import torch
from tqdm import tqdm
import torch.nn.functional as F
import time
import torchvision.models as models
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

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
    for (X,y) in data:
        all+=1
        y_pred = model(X.unsqueeze(0))
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
    TRAIN_PATH = "data\ChnSentiCorp_htl_all\\train_1600+1600.csv"
    TEST_PATH = "data\ChnSentiCorp_htl_all\\test_800+800.csv"
    BATCH_SIZE = 64
    EPOCH_NUM = 10
    LEARNING_RATE = 0.01
    PADDING_LEN = 128
    DROUP_OUT = 0.7
    NAME = f'transformer_bs{BATCH_SIZE}_lr{LEARNING_RATE}_en{EPOCH_NUM}_adam_d{DROUP_OUT}_pl{PADDING_LEN}'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    currentTime = datetime.now().strftime('%b%d_%H-%M-%S')
    writer = SummaryWriter(f"runs/{NAME}_{currentTime}/")

    # 加载数据
    eb=oneHotEmbedding("data\ChnSentiCorp_htl_all\ChnSentiCorp_htl_all.csv",PADDING_LEN)
    train_data = ChSentiDataSet(TRAIN_PATH,eb)
    test_data = ChSentiDataSet(TEST_PATH,eb)

    # train data
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    start = time.time()
    # 创建模型
    # model = MLPmodelV1()
    # model = MLPmodelV2()
    # model = textCNNv1((PADDING_LEN, len(eb)))
    # model = LSTMv1(len(eb), 32, PADDING_LEN, dropout=0.6)
    model = GRUv1(len(eb), 32, PADDING_LEN, dropout=0.6)
    model = transformerv1(PADDING_LEN, len(eb))

    # 优化器
    # optimizer = myOptimSimple(model.parameters(), lr=LEARNING_RATE)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best = 0
    for epoch in range(EPOCH_NUM):
        model.train()
        totloss = 0.0
        print(f"Epoch: {epoch+1}/{EPOCH_NUM}")
        # for (X, y_std) in tqdm(train_dataloader):
        for (X, y_std) in train_dataloader:
            y_pred = model(X)
            loss = F.nll_loss(y_pred, y_std)
            totloss+=loss
            optimizer.zero_grad() # 这个是梯度置零 添加set_to_none可以置为None，会占用更小的内存，但是会出事
            loss.backward()
            optimizer.step()

        print(f"loss:{totloss}")
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

    end = time.time()
    print(f"time:{end-start}\n")
     
    writer.close()