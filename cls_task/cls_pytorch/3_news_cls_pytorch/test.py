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

def main(test, pretrained_file, batch_size=1):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # bert_tokenizer = BertTokenizer.from_pretrained("chinese_wwm_ext_L-12_H-768_A-12/vocab.txt", do_lower_case=True)
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
    print(20 * "=", " Preparing for testing ", 20 * "=")
    if platform == "linux" or platform == "linux2":
        checkpoint = torch.load(pretrained_file+'/best.pth.tar')
    else:
        checkpoint = torch.load(pretrained_file+'/best.pth.tar', map_location=device)
    # Retrieving model parameters from checkpoint.
    print("\t* Loading test data...")    
    test_data = DataPrecessForSentence(bert_tokenizer, test,pred=True)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    print("\t* Building model...")
    config = Config()
    model = BertModelTest(pretrained_file+'/config.json',config).to(device)
    model.load_state_dict(checkpoint["model"])
    print(20 * "=", " Testing BERT model on device: {} ".format(device), 20 * "=")
    result = predict(model, test_loader, device)
    print(result)

    # batch_time, total_time, accuracy, auc = test(model, test_loader)
    # print("\n-> Average batch processing time: {:.4f}s, total test time: {:.4f}s, accuracy: {:.4f}%, auc: {:.4f}\n".format(batch_time, total_time, (accuracy*100), auc))


if __name__ == "__main__":
    text = ['一4个月大女婴疑遭生母从5楼扔下，曾两次送医又被其父接回，其父签署了《放弃治疗协议书》。'
            '12月8日，记者从友谊街道办事处获悉，7日晚，坠楼女婴被送去河北省儿童医院接受治疗，由社区领导'
            '及工作人员实时陪护，目前孩子状态较为稳定。',
            '美东时间周一，美股三大指数涨跌互现。截至当日收盘，纳指报12519.95点，续创收盘历史新高，涨幅0.45%；'
            '道指报30069.79点，跌0.49%；标普500指数报3691点，微跌0.19%.个股方面，特斯拉股价涨逾7%，报641.76美元，'
            '公司总市值首次突破6000亿美元，达到了6083亿美元（约3.9万亿元人民币），市值一天就增长了404亿美元（约2643亿元人民币）。'
            '同期丰田汽车的市值为1931亿美元。也就是说，特斯拉的市值已超过3个丰田汽车。',
            '近日有监控画面显示，在一所幼儿园内的午休时间，有孩子不好好睡觉，一位阿姨径直坐在孩子身上开始玩手机，'
            '长达三分钟不起身。视频画面一经曝光立即引发网民热议。',
            '12月7日，G373次高铁唐山到长春途中，一中年女子与母亲发生口角。女子不断辱骂母亲：“我咋不死你呢！死了得了！”'
            '据悉，母亲还表示自己曾被女子伤。一旁乘客上前劝阻，被骂人女子怼：“你们管吧！给你们，你们管。”随后女子扬长而去。'
            '文明提醒：公共场所请注意公共秩序！家人之间请友善沟通！']
    main(text, "./models")
    # main("./data/ChineseSTS-master/simtrain.csv", "./models")

