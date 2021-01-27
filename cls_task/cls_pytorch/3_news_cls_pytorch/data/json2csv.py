#!/usr/bin/python3
# -*- coding: utf-8 -*-

# 导入CSV安装包
import csv
import json
# 1. 创建文件对象
f1 = open('only_title_ccf_pytorch.csv','w',encoding='utf-8',newline='')

# 2. 基于文件对象构建 csv写入对象
csv_writer = csv.writer(f1)

# 3. 构建列表头
csv_writer.writerow(["text","label"])

# 4. 写入csv文件内容

with open('/media/zhihao/DATA/code/nlp/network_consensus/data/2019年CCF比赛互联网新闻情感分析数据集/only_title_ccf2019_cls.txt','r') as f:
    all = json.load(f)
    # all_lst = []
    for i in all:
        single = []
        single.append(i['content'])
        if i['label'] == "negtive":
            single.append('2')
        if i['label'] == "neutral":
            single.append('1')
        if i['label'] == "positive":
            single.append('0')
        csv_writer.writerow(single)

# 5. 关闭文件
f1.close()
