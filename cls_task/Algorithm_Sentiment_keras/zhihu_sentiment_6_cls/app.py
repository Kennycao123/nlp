from util.data_util import db
from flask import Flask,request

from single_test import *
import json
app = Flask(__name__)

model = load_model()

@app.route('/sentiment', methods=['POST'])
def single_sentiment_analysis_1():
    if request.method == 'POST':
        content = request.json 
        text = content["text"]   #text 是字段
        print("输入的句子:",text)
        test_D,test_model_pred = data_prepare(text)
        sentiment = predict(model,test_D,test_model_pred)
        res = {}
        res['sentiment'] = sentiment
        print("输出的结果:",res)
        return json.dumps(res)
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=20001)
