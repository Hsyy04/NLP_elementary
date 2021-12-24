from data import embedding, ChSentiDataSet
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch
from tqdm import tqdm

# version1
# 使用全连接层实现
class MLPmodelV1(nn.Module):
    def __init__(self, inputsize=350) -> None:
        super().__init__()
        self.hidden1 = nn.Linear(inputsize, 64, bias=True)
        self.hidden2 = nn.Linear(64, 16, bias=True)
        self.output = nn.Linear(16, 2, bias=True)

    def forward(self, input):
        print(input)
        x = F.relu(self.hidden1(input.float()))
        x = F.relu(self.hidden2(x))
        x = F.log_softmax(self.output(x),dim=-1)
        return x

class MLPmodelV2(nn.Module):
    def __init__(self, inputsize=350) -> None:
        super().__init__()
        self.w1 = torch.rand((inputsize,64),requires_grad=True)
        self.b1 = torch.rand((64,),requires_grad=True)
        self.w2 = torch.rand((64,16),requires_grad=True)
        self.b2 = torch.rand((16,),requires_grad=True)
        self.wo = torch.rand((16,2),requires_grad=True)

    def forward(self, input):
        x = input @ self.w1 + self.b1
        # x = F.relu
        return x

def valid(data, model, phrase):
    currect = 0
    all = 0
    for X,y in data:
        all+=1
        y_pred = model(X)
        y_pred = torch.argmax(y_pred,dim=-1)
        if y_pred == y:
            currect+=1
    print(phrase)
    print(f"acc: {currect}/{all}:({float(currect)*100.0/float(all)}%)")

if __name__ == "__main__":
    # const valuables
    BATCH_SIZE = 64
    TRAIN_PATH = "data\ChnSentiCorp_htl_all\\train_1600+1600.csv"
    TEST_PATH = "data\ChnSentiCorp_htl_all\\test_800+800.csv"
    EPOCH_NUM = 10
    LEARNING_RATE = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    # 加载数据
    eb=embedding("data\ChnSentiCorp_htl_all\\train_1600+1600.csv")
    train_data = ChSentiDataSet(TRAIN_PATH,eb)
    test_data = ChSentiDataSet(TEST_PATH,eb)

    # train data
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # 创建模型
    model = MLPmodelV1()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCH_NUM):
        totloss = 0.0
        print(f"Epoch: {epoch+1}/{EPOCH_NUM}")
        for (X, y_std) in tqdm(train_dataloader):
            y_pred = model(X)
            loss = F.nll_loss(y_pred, y_std)
            totloss+=loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"loss:{totloss}")
        valid(train_data, model,"vaild")
        valid(test_data, model,"test")
        print('\n')
        