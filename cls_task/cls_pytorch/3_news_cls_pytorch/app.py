from flask import Flask,request

from single_test import *
from utils import predict
import json
app = Flask(__name__)

model,device = load_model()

@app.route('/news', methods=['POST'])
def single_sentiment_analysis_1():
    if request.method == 'POST':
        content = request.json 
        text = content["text"]   #text 是字段
        # print("输入的文本:",text)
        print(len(text))
        if len(text) > 200:
            text = abstract(text)
            print("抽取摘要后输入的文本:",text)
        test_loader = data_prepare(text)
        sentiment = predict(model,test_loader,device)[0]
        print(sentiment)
        res = {}
        res['news_sentiment'] = sentiment
        print("输出的结果:",res)
        return json.dumps(res)
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=20003)