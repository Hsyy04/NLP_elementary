from xml.dom.minidom import Document
import tokenizers
from transformers import BertTokenizer
from torch.utils.data import Dataset
import pandas as pd
import jieba 
'''
1. create the dictionary
    -- use data/bert_min/vocab.txt from Internet 
2. create the pretrain data
'''

class BertDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        Document = pd.read_csv('data/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv',keep_default_na=False, header=0)

if __name__ == "__main__":
    dataset = BertDataset()
    print(dataset.Document)
