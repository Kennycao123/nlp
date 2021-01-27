# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 13:30:07 2020

@author: zhaog
"""
import torch
from sys import platform
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import BertModelTest,Config
from utils import test,predict
from data import DataPrecessForSentence

from snownlp import SnowNLP
import re  

def data_prepare(text):
    single = []
    single.append(text)
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
    test_data = DataPrecessForSentence(bert_tokenizer, single,pred=True)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=1)
    return test_loader
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if platform == "linux" or platform == "linux2":
        checkpoint = torch.load('./models/best.pth.tar')
    else:
        checkpoint = torch.load('./models/best.pth.tar', map_location=device)
    print("\t* Building model...")
    config = Config()
    model = BertModelTest('./models/config.json',config).to(device)
    model.load_state_dict(checkpoint["model"])
    print(20 * "=", " Testing BERT model on device: {} ".format(device), 20 * "=")
    return model,device

def abstract(news):
    sentence = re.split(r'。|\n|\?|\？|！|!|……',news)
    if len(sentence) >5:
        s = SnowNLP(news).summary(5)
    else:
        s = SnowNLP(news).summary(len(sentence))
    news = ",".join(s)
    return news

if __name__ == "__main__":

    model,device = load_model()
    text = '这件事情真是太恶劣了，居然发生这样的车祸！！！'
    if len(text) > 200:
        text = abstract(text)
    print(text)
    test_loader = data_prepare(text)
    res = predict(model,test_loader,device)[0]
    print(res)