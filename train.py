from torch.optim.optimizer import Optimizer
from data import embedding, ChSentiDataSet
from MLP.MLP import MLPmodelV1, MLPmodelV2
from torch.utils.data import DataLoader,random_split
import torch
from tqdm import tqdm
import torch.nn.functional as F
import time
import torchvision.models as models
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
    currect = 0
    all = 0
    loss = 0.0
    for (X,y) in data:
        all+=1
        y_pred = model(X)
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
    EPOCH_NUM = 2
    LEARNING_RATE = 0.01
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    writer = SummaryWriter()

    # 加载数据
    eb=embedding("data\ChnSentiCorp_htl_all\\train_1600+1600.csv")
    train_data = ChSentiDataSet(TRAIN_PATH,eb)
    test_data = ChSentiDataSet(TEST_PATH,eb)

    # train data
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    start = time.time()
    # 创建模型
    model = MLPmodelV1()
    model = MLPmodelV2()
    # 优化器
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    optimizer = myOptimSimple(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCH_NUM):
        test = 1
        totloss = 0.0
        print(f"Epoch: {epoch+1}/{EPOCH_NUM}")
        for (X, y_std) in tqdm(train_dataloader):
        # for (X, y_std) in train_dataloader:
            test+=1
            if test == 2:
                print(X)
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
        writer.add_scalars('loss', dic, epoch)
        dic = {'val':val_acc,'test':test_acc}
        writer.add_scalars('acc', dic, epoch)

        print('\n')

    end = time.time()
    print(f"time:{end-start}\n")
    torch.save(model, f'model/bs{BATCH_SIZE}_lr{LEARNING_RATE}_en{EPOCH_NUM}.pth')  
    writer.close()