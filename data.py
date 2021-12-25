import pandas as pd
from torch._C import set_num_interop_threads
# from torch._C import float32
import numpy as np
import torch.nn.functional as F
from torch.utils import data
import os
import torch
from torch.utils.data import DataLoader,Dataset
import jieba


class ChSentiDataSet(Dataset):
    def __init__(self, data_path, embedding) -> None:
        # 由于数据比较少, 我们直接一次加载进来就好.
        # label, review
        self.data_all = pd.read_csv(data_path)
        self.embedding = embedding
        super().__init__()
        pass

    def __len__(self):
        # 返回数据集大小
        return self.data_all.shape[0]

    def __getitem__(self, index):
        # 返回下标为index的数据项
        review = self.data_all.iloc[index,1]
        review = self.embedding.toTensor(review)
        label =int(self.data_all.iloc[index,0])
        # label = torch.zeros(2, dtype=torch.float).scatter_(dim=0, index=torch.tensor(label), value=1)
        return review, label

class embedding:
    # 不太了解embedding, 先使用所有训练数据中的句子拆出的词表, 把出现次数少(<100)的词语视为other
    # 拆出的词表需要去掉停止词
    def __init__(self, all_sentence_path) -> None:
        # 读取所有句子
        all_sentence = pd.read_csv(all_sentence_path,header=0,names=['label','sentence'],keep_default_na=False)
        #读取停止词
        self.stop_words = pd.read_csv("data/stop_words.csv",header=None,sep='!')

        # 计算词频
        # FIXME: 性能有待提升
        self.word_dict = {}
        for sentence in all_sentence['sentence']:
            words = jieba.lcut(sentence)
            for wd in words:
                self.word_dict[wd]= self.word_dict[wd]+1 if self.word_dict.__contains__(wd) else 1
            
        # 得到所有单词
        self.words = list(self.word_dict.keys())

        # 删除停止词和低频词
        for wd in self.word_dict:
            if (wd in self.stop_words) or (self.word_dict[wd]<=100):
                self.words.remove(wd)

        # 添加other
        self.words.append("other类")

    def __len__(self):
        return len(self.words)

    def toTensor(self, sentence):
        # 得到一个句子的向量化结果
        try:
            st_words = jieba.lcut(sentence)
        except:
            # print(sentence)
            st_words = [' ']
        wd_dict = dict(zip(self.words, np.zeros(len(self.words))))
        for wd in st_words:
            # if wd_dict.__contains__(wd):
            if wd in self.stop_words:
                continue
            if wd in self.words:
                wd_dict[wd] = wd_dict[wd]+1
            else:
                wd_dict["other类"]+=1
        
        ret = torch.tensor(list(wd_dict.values()),dtype=float)
        ret = F.softmax(ret,dim=-1)
        # assert(False)
        return ret
        

if __name__ == "__main__":
    train_data = ChSentiDataSet("data\ChnSentiCorp_htl_all\\train_1600+1600.csv")
    # train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    # train_features, train_labels = next(iter(train_dataloader))
    # embd = embedding("data\ChnSentiCorp_htl_all\\train_1600+1600.csv")
    # print(len(embd))
    # print(embd.toTensor("黑店,黑店,绝对黑店.黑店,黑店,绝对黑店.黑店,黑店,绝对黑店."))