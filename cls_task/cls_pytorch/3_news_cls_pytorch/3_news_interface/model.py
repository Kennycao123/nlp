# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 21:06:25 2020

@author: zhaog
"""
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertForSequenceClassification, BertConfig
from transformers import BertModel, BertTokenizer

class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'bert'
        self.class_list = ['正面','中性','负面']   # 类别名单
        self.num_classes = len(self.class_list)  # 类别数
        self.hidden_size = 768


class Bert_Model(nn.Module):
    def __init__(self,config):
        super(Bert_Model, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.device = torch.device("cuda")
        for param in self.bert.parameters():
            param.requires_grad = True  # 每个参数都要 求梯度
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, batch_seqs, batch_seq_masks,batch_seq_segments,labels):
        _, output = self.bert(batch_seqs, attention_mask = batch_seq_masks)
        logits = self.fc(output)
        loss_fn = CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        probabilities = nn.functional.softmax(logits, dim=-1)

        return loss, logits, probabilities
    
    
class BertModelTest(nn.Module):
    def __init__(self,model_path,conf):
        super(BertModelTest, self).__init__()
        config = BertConfig.from_pretrained(model_path)
        # self.bert = BertForSequenceClassification(config)  # /bert_pretrain/
        self.bert = BertModel(config)
        self.fc = nn.Linear(conf.hidden_size, conf.num_classes )
        self.device = torch.device("cuda")

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        _, output = self.bert(batch_seqs, attention_mask = batch_seq_masks)
        logits = self.fc(output)
        probabilities = nn.functional.softmax(logits, dim=-1)
        loss_fn = CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        return loss, logits, probabilities