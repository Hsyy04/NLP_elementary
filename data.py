from lib2to3.pgen2 import token
from turtle import pos
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils import data
import os
import torch
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import jieba
from transformers import BertTokenizer

class ChSentiDataSet(Dataset):
    def __init__(self, data_path, embedding) -> None:
        # 由于数据比较少, 我们直接一次加载进来就好.
        # label, review
        self.data_all = pd.read_csv(data_path, keep_default_na=False, header=0)
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
        return review, label

class embedding:
    # 不太了解embedding, 先使用所有训练数据中的句子拆出的词表, 把出现次数少(<100)的词语视为other 
    # 拆出的词表需要去掉停止词
    def __init__(self, all_sentence_path,minfr=100) -> None:
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
            if self.word_dict[wd]<=minfr:
                self.words.remove(wd)

        # 添加other
        self.words.append("@other")
        self.words.append("@pad")
        self.words.append("@cls")
        self.words.append("@sep")

    def __len__(self):
        return len(self.words)

    def toTensor(self, sentence):
        # 得到一个句子的向量化结果
        # 词袋
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
                wd_dict["@other"]+=1
        
        ret = torch.tensor(list(wd_dict.values()),dtype=float)
        ret = F.softmax(ret,dim=-1)
        return ret

class oneHotEmbedding(embedding):
    def __init__(self, all_sentence_path, length, minfr=100) -> None:
        super().__init__(all_sentence_path, minfr)
        # 得到词典
        self.wordsindex = dict((_,i) for i,_ in enumerate(self.words))

        self.length = length

    def toTensor(self, sentence):
        sent_words = jieba.lcut(sentence)
        sent_index = []
        # 把词语转换维字典对应的序号
        for word in sent_words:
            if self.wordsindex.__contains__(word):
                sent_index.append(self.wordsindex[word]) # 已存在的
            else:
                sent_index.append(self.wordsindex['@other']) # unk
        while len(sent_index) < self.length:  # padding
            sent_index.append(self.wordsindex['@pad'])

        # 将id转换为独热码
        id = torch.tensor(sent_index[:self.length]).reshape(self.length,1) 
        sent_tensor = torch.zeros((self.length,len(self.words)), dtype=torch.float).scatter_(dim=1, index=id, value=1)

        return sent_tensor.unsqueeze(0) # 为了后面卷积方便调用接口, 因此加一层channel维

class indexDictEmbedding(oneHotEmbedding):
    def __init__(self, all_sentence_path, length, minfr=100) -> None:
        super().__init__(all_sentence_path, length, minfr)
    
    def toTensor(self, sentence):
        sent_words = jieba.lcut(sentence)
        sent_index = []
        # 把词语转换维字典对应的序号
        for word in sent_words:
            if self.wordsindex.__contains__(word):
                sent_index.append(self.wordsindex[word]) # 已存在的
            else:
                sent_index.append(self.wordsindex['@other']) # unk
        while len(sent_index) < self.length:  # padding
            sent_index.append(self.wordsindex['@pad'])

        # 得到句子中每个单词的id
        id = torch.tensor(sent_index[:self.length])

        return id.unsqueeze(0) # 为了后面卷积方便调用接口, 因此加一层channel维

class bertEmbedding(embedding):
    def __init__(self, all_sentence_path, length, minfr=100) -> None:
        super().__init__(all_sentence_path, minfr)
        self.length = length
        # 使用的预训练模型：https://github.com/ymcui/Chinese-BERT-wwm
        self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm',padding=True, truncation=True, return_tensors="pt")
    
    def toTensor(self, sentence):
        token_words = self.tokenizer.encode_plus(sentence, max_length=self.length, padding='max_length',truncation=True)
        # return torch.tensor([token_words['input_ids'], token_words['token_type_ids'], token_words['attention_mask']])
        return torch.tensor(token_words['input_ids'])

class bertEmbeddingv1(embedding):
    def __init__(self, all_sentence_path, length, minfr=100) -> None:
        super().__init__(all_sentence_path, minfr)
        # 得到词典
        self.wordsindex = dict((_,i) for i,_ in enumerate(self.words))
        self.length = length
    
    def toTensor(self, sentence):
        sent_words = jieba.lcut(sentence)
        sent_index = []
        # 把词语转换维字典对应的序号
        sent_index.append(self.wordsindex['@cls'])
        for word in sent_words:
            if self.wordsindex.__contains__(word):
                sent_index.append(self.wordsindex[word]) # 已存在的
            else:
                sent_index.append(self.wordsindex['@other']) # unk
        sent_index.append(self.wordsindex['@sep'])

        while len(sent_index) < self.length:  # padding
            sent_index.append(self.wordsindex['@pad'])

        # 得到句子中每个单词的id
        input_ids = torch.tensor(sent_index[:self.length])
        attention_mask = torch.tensor([i!=self.wordsindex['@pad'] for i in sent_index])
        token_type_ids = torch.tensor([0]*self.length)
        position_ids = torch.tensor([i for i in range(self.length)])
        return torch.tensor(input_ids, attention_mask, token_type_ids, position_ids)

class corpusInfo:
    def __init__(self, path) -> None:
        self.data = pd.read_csv(path, keep_default_na=False, header=0, names=['label','sentence'])

    def maxWords(self):
        max_cnt = 0
        max_sent =''
        for sent in self.data['sentence']:
            word_sent = jieba.lcut(sent)
            if max_cnt < len(word_sent):
                max_cnt = len(word_sent)
                max_sent = sent
        return (max_sent, max_cnt)
    
    def minWords(self):
        min_cnt = 1000000
        min_sent =''
        for sent in self.data['sentence']:
            word_sent = jieba.lcut(sent)
            if min_cnt > len(word_sent) and len(word_sent) != 0:
                min_cnt = len(word_sent)
                min_sent = sent
        return (min_sent, min_cnt)

    def meanWords(self):
        mean_cnt = 0.0
        for sent in self.data['sentence']:
            word_sent = jieba.lcut(sent)
            mean_cnt += len(word_sent)
        return mean_cnt/3200.0

    def histWords(self):
        cnt = []
        for sent in self.data['sentence']:
            cnt.append(len(jieba.lcut(sent)))
        cnt.sort()
        plt.hist(cnt,bins=300)
        plt.show()
        print(cnt[round(len(cnt)*0.9)])

if __name__ == "__main__":
    # info = corpusInfo("data\ChnSentiCorP_htl_all\ChnSentiCorp_htl_all.csv")
    # print(info.maxWords())
    embedding = bertEmbedding("data\ChnSentiCorP_htl_all\ChnSentiCorp_htl_all.csv", 128)
    train_data = ChSentiDataSet("data\ChnSentiCorp_htl_all\\train_1600+1600.csv", embedding)
    print(train_data.__getitem__(1))
