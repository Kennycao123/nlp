1.目录总览    

两个文件夹分别是知乎数据情感六分类和新闻数据情感三分类的模型文件夹   

2.两个文件夹区别     

zhihu_sentiment文件夹主要针对知乎短文本情感分类,对文本内容直接分类。news_sentiment文件夹下的模型，对输入进来的新闻数据首先进行长度判断，大于200的就先进行摘要提取，然后输入分类模型分类。

2.需要的环境：

```
python:3.6
tensorflow >= 2.2.0  
keras:2.4.3  
keras_bert:0.86.0
flask
snownlp
```

2.中文预训练模型

[哈工大中文预训练模型](https://github.com/ymcui/Chinese-BERT-wwm)

3.数据集训练

数据格式转换成json,具体参考zhihu_sentiment_6_cls里面usual_data文件夹下训练和测试数据集格式

4.训练好的模型默认保存在bert_dump里面

5.单例测试命令

`python3 single_test.py`

6.启动后台服务

`python3 app.py`

7.发送请求样例

`python3 request.py`

8.显存限制

两个模型测试时候均限制运行时候占用显存为2G（模型运行所需最低显存）.可修改single_test.py里面19-21行相应代码改变占用显存大小