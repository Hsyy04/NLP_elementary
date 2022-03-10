from venv import create
from transformers import BertTokenizer
from torch.utils.data import Dataset
import pandas as pd
import torch
import random

import re
def cut_sent(para):
    # 中文分句小工具
    para = re.sub('([~;；.。,，！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")

'''
1. create the dictionary
    -- use data/bert_min/vocab.txt from Internet 
2. create the pretrain data
'''

class BertDataset(Dataset):
    def __init__(self, max_len=256) -> None:
        super().__init__()
        self.tokenizer = BertTokenizer('data/bert_min/vocab.txt')
        # 从评论数据中生成document, 删除了短评.
        self.all_document = pd.read_csv('data/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv',header=0,names=['label','sentence'],keep_default_na=False)['sentence'].values.tolist()
        for i in self.all_document:
            if len(i) < 384:
                self.all_document.remove(i)
            elif len(cut_sent(i))<=2:
                self.all_document.remove(i)

        self.data_tokens=[]
        for i in range(len(self.all_document)):
            self.data_tokens.extend(self.create_instances(i,max_len)) # 这里是先构造句子对
        self.data_inputs=[]
        for tk in self.data_tokens:
            self.data_inputs.append(self.create_input(tk,max_len=max_len))
        self.max_len=max_len
        
    def create_input(self, token, mask_len=30, mask_prob=0.15, max_len=256):
        '''
        对于传入的token, 要生成输入
        输入包括原id, 掩码之后的id, padding掩码, 被掩码掩去的pos
        '''
        for i in range(len(token['tokens']), max_len):        # 添加padding
            token['tokens'].append('[PAD]')
            token['segment'].append(1)
        all_id = self.tokenizer.convert_tokens_to_ids(token['tokens'])  # 转换原token到id
        mask_pos=[]
        for pos,id in enumerate(all_id):            # 先找到所有能够mask的位置
            if (id>=100 and id<=103) or id == 0:
                continue
            mask_pos.append(pos)
        mask_num = min(mask_len, max(1,int(len(mask_pos)*mask_prob)))
        random.shuffle(mask_pos)
        mask_pos = mask_pos[:mask_num]
        while mask_num< mask_len : 
            mask_pos.append(-1)
            mask_num+=1
        
        mask_id = []                    # 生成掩码的id
        for pos,id in enumerate(all_id):
            if id>=100 and id<=103:
                mask_id.append(id)
            else:
                if pos in mask_pos:
                    rd = random.random()
                    if rd<0.8 :
                        mask_id.append(103)
                    elif rd<0.9:
                        mask_id.append(id)
                    else:
                        tkid = random.randint(0,self.tokenizer.vocab_size-1)
                        while tkid == id : tkid = random.randint(0,self.tokenizer.vocab_size-1)
                        mask_id.append(tkid)
                else:
                    mask_id.append(id)
        mask_padding=[0 if i>0 else 1 for i in all_id] # 参考TransformerEncoder的接口
        # create
        ret = {'token_ids':all_id, 'mask_ids':mask_id, 'mask_padding':mask_padding,'mask_pos': mask_pos}
        #      所有的token           被bert mask掉后的整句token           padding           mask掉的在原句中的位置
        return ret
        
    def create_instances(self, index, max_len=256):
        '''
        对一个文档构造句子对,返回构造的句子对数组
        句子A是原文档中的几句, 句子B有50%的概率是A的下一句, 50%的概率是别的文档的随机一句
        为了使得padding尽量的少, 那么句子就要尽量地长.
        '''
        doc = cut_sent(self.all_document[index]) # 将文档分句
        i = 0
        cur_sent = []
        cur_len = 0
        used_end=0
        tokens=[]
        while i < len(doc):
            if cur_len + len(doc[i])< max_len-3:         # 如果长度允许可以多加一点句子
                cur_sent.append(doc[i])
                cur_len += len(doc[i])
                i+=1
                if i != len(doc):
                    continue
            # 此时已经超出了长度限制, 从cur_sent中选择一句末尾截断
            tokena=''
            if len(cur_sent)==0:
                # 有一个句子直接大于max_len了
                i+=1
                used_end+=1
                continue
            a_end = 1 if len(cur_sent)<=1 else random.randint(1,len(cur_sent))
            for tem in range(a_end):
                tokena+=cur_sent[tem]
                used_end+=1

            # 选择句子B
            tag=1
            tokenb=''
            if random.random()<0.5:         # 选择其他文档中的
                tag=0
                fake_doc = random.randint(0,len(self.all_document)-1)
                while fake_doc == index: fake_doc = random.randint(0,len(self.all_document)-1)
                fake_doc = cut_sent(self.all_document[fake_doc])
                start = random.randint(0,len(fake_doc))     # 随机选择一个开始的位置

                for id in range(start,len(fake_doc)):       # 添加句子B直到不能在加
                    if len(tokena)+len(tokenb)+len(fake_doc[id]) > max_len-3:
                        break
                    tokenb+=fake_doc[id]    
            else:                           # 选择真实文档中的
                tag=1
                start = used_end
                for id in range(used_end, len(doc)):   # 在本文档中添加真实的句子
                    if len(tokena)+len(tokenb)+len(doc[id]) > max_len-3:
                        break
                    tokenb+=doc[id]
                    used_end+=1
            
            # create one
            if len(tokenb)!=0:
                token=['[CLS]']
                seg = [0]
                for tk in self.tokenizer.tokenize(tokena):
                    token.append(tk)
                    seg.append(0)
                seg.append(0)
                token.append('[SEP]')
                for tk in self.tokenizer.tokenize(tokenb):
                    token.append(tk)
                    seg.append(1)
                seg.append(1)
                token.append('[SEP]')
                tokens.append({'tokens':token,'segment':seg,'target':tag})
            # update
            cur_len = 0
            cur_sent = []
            i = used_end 

        return tokens

    def __len__(self):
        return len(self.data_inputs)

    def __getitem__(self, index):
       '''
       X: input_ids, 
            pos_ids,  
            segment_ids, 
            mask_vec,
       X1:all_ids,
       Y1: mask_pos     mask掉的词的位置
       Y2: tag
       '''
       input_ids = self.data_inputs[index]['mask_ids']
       pos_ids = [i for i in range(self.max_len)]
       segment_ids = self.data_tokens[index]['segment']
       mask_vec = self.data_inputs[index]['mask_padding']
       all_ids = torch.Tensor(self.data_inputs[index]['token_ids'])
       X = torch.Tensor([input_ids, pos_ids, segment_ids, mask_vec]).to(int)
       Y1 = torch.Tensor(self.data_inputs[index]['mask_pos']).to(int)
       Y2 = self.data_tokens[index]['target']
       return X, all_ids, Y1, Y2

if __name__ == "__main__":
    dataset = BertDataset()
    test = dataset.__getitem__(1)
    print(test[0][-1].shape)
    