import torch
from sys import platform
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import BertModelTest
from utils import predict
from data import DataPrecessForSentence
import pandas as pd
from datetime import datetime


def creat_testdata(test_file):
    test = pd.read_csv(test_file)
    raw_test_list = test['abstract_news'].values
    return raw_test_list

def load_model( pretrained_file ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
    print(5 * "=", " Preparing for predict_result ", 5 * "=")
    if platform == "linux" or platform == "linux2":
        checkpoint = torch.load(pretrained_file+'/best.pth.tar')
    else:
        checkpoint = torch.load(pretrained_file+'/best.pth.tar', map_location=device)
    print("\t* Building model...")
    model = BertModelTest(pretrained_file + '/config.json').to(device)
    model.load_state_dict(checkpoint["model"])
    # Retrieving model parameters from checkpoint.
    return model, bert_tokenizer, device

def predict_result(model,bert_tokenizer,device,test_list,batch_size=1):
    print("\t* Loading test data...")
    test_data = DataPrecessForSentence(bert_tokenizer, test_list, pred=True)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    result = predict(model, test_loader, device)
    return result
    # text_result = []
    # for i, j in enumerate(test_list):
    #     text_result.append([j[0], j[1], '相似' if torch.argmax(result[i][0]) == 0 else '不相似'])

def main(test_file):
    raw_test_list = creat_testdata(test_file)
    model, bert_tokenizer, device = load_model("./models")
    predict_text_result = predict_result(model, bert_tokenizer,device,raw_test_list)
    names=['text','predict']
    text_predict = pd.DataFrame(columns=names,data=zip(raw_test_list, predict_text_result))#数据有yi列，列名分别为abstract_news
    text_predict.to_csv('text_predict.csv')


if __name__ == "__main__":
    time1 = datetime.now()

    test_file = "./data/abstract_news.csv"
    main(test_file)

    time2 = datetime.now()
    print('程序运行的时间:', time2 - time1)