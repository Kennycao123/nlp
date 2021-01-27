import pandas as pd
import codecs, gc
import numpy as np
from sklearn.model_selection import KFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.metrics import top_k_categorical_accuracy
from keras.layers import *
from keras.callbacks import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
import json
import re
from snownlp import SnowNLP
 
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
#assert len(gpus) > 0, "Not enough GPU hardware devices available"
if len(gpus)>0:
    tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

maxlen = 200
label_list = ['positive', 'neutral', 'negtive']

config_path = 'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_config.json'
checkpoint_path = 'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_model.ckpt'
dict_path = 'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/vocab.txt'

#将词表中的词编号转换为字典
token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
#重写tokenizer        
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # 用[unused1]来表示空格类字符
            else:
                R.append('[UNK]')  # 不在列表的字符用[UNK]表示
        return R
tokenizer = OurTokenizer(token_dict)
#让每条文本的长度相同，用0填充
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])
#data_generator只是一种为了节约内存的数据方式
class data_generator:
    def __init__(self, data, batch_size=8, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
 
    def __len__(self):
        return self.steps
 
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
 
            if self.shuffle:
                np.random.shuffle(idxs)
            
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y[:, 0, :]
                    [X1, X2, Y] = [], [], []
#计算top-k正确率,当预测值的前k个值中存在目标类别即认为预测正确                 
def acc_top2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)
#bert模型设置
def build_bert(nclass):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)  #加载预训练模型
    for l in bert_model.layers:
        l.trainable = True
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x) # 取出[CLS]对应的向量用来做分类
    p = Dense(nclass, activation='softmax')(x)
    model = Model([x1_in, x2_in], p)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(1e-5),    #用足够小的学习率
                  metrics=['accuracy', acc_top2])
    print(model.summary())
    return model

def data_prepare(text):
    single = []
    single.append((text, to_categorical(0, 3)))
    single = np.array(single)
    test_D = data_generator(single, batch_size=1,shuffle=False)
    test_model_pred = np.zeros((len(single), 3))
    return test_D,test_model_pred
def load_model():
    model = build_bert(3)
    # model.load_weights('./bert_dump/' + str(i) + '.hdf5')
    model.load_weights('./bert_dump/' + '0' + '.hdf5')
    return model
def predict(model,test_D,test_model_pred):
    test_model_pred += model.predict_generator(test_D.__iter__(), steps=len(test_D), verbose=1)
    del model
    gc.collect()   #清理内存
    K.clear_session()   #clear_session就是清除一个session
    test_pred = [np.argmax(x) for x in test_model_pred]
    res = label_list[test_pred[0]]
    return res

def abstract(news):
    sentence = re.split(r'。|\n|\?|\？|！|!|……',news)
    if len(sentence) >5:
        s = SnowNLP(news).summary(5)
    else:
        s = SnowNLP(news).summary(len(sentence))
    news = ",".join(s)
    return news

if __name__ == "__main__":
    model = load_model()
#     text = "今年1月房企合同销售金额遭遇寒潮，但碧桂园控股（02007.HK）董事局主席杨国强对于国内经济形势以及房地产行业的走向，似乎有着积极的信念。\
# 杨国强在年初的一个会议上提出：当前中国经济形势复杂，但要对经济稳定增长、长期向好抱有坚定信心。“过去不代表现在，现在也不代表未来。\
# ”2018年，碧桂园以7286.9亿蝉联销售金额榜榜首，其中权益销售金额为5203.6亿，较2017年同期的3961.1亿同比稳健增长30%。\
# 这是碧桂园在上一轮房地产上涨周期之前提早布局的结果，但随着2018年国家层面提出“房住不炒”的高层定调之后，市场的预期已经发生改变，买卖双方正回归理性，大多数房企对于行业的前景也变得不再盲目乐观。\
# 碧桂园2月3日公告称，2019年1月该集团共实现归属公司股东权益的合同销售金额约人民币330.7亿元，与2018年同期相比大幅下滑52.2%。\
# 碧桂园早在2018年中期便提出“提质控速”之全新发展思路，主动切换车道以适应新的市场节奏。如今，杨国强依然看好房地产行业在中国城镇化进程中的发展机会。\
# 杨国强认为：“中国的城镇化、现代化是不可阻挡的，房地产行业每年起码有10万亿的市场。目前行业出现波动是正常的，淘汰一些竞争力较弱的企业对于行业健康发展是长远有利的。\
# ”对于处于行业头部地位的碧桂园来说，更大的市场份额依然值得憧憬。而当下市场处于调整周期之际，杨国强也对风险作出考量。他要求公司要“财务稳健，接下来要以销定产、量入为出。\
# 一定要有质量的发展，一定要科学地去谋划，提升全周期竞争力，精准地投入，实现长期效益和短期效益的有机结合。\
# 同时，这家公司在巩固房地产开发核心业务的同时，已然精心布局了机器人、农业等延伸业务，并试图让这些业务之间产生协同效益，有效调用现有资源，提高企业资源使用效率。\
# 杨国强表示：“去年我们农业公司和机器人公司的框架已经搭起来了，发展的思路也有了，接下来期待他们的精彩表演。未来我们是三个重点，地产、农业、机器人。\
# ”如此看来，2019年碧桂园的三个业态将同步发展，协同发展，去全力提升集团的总体竞争实力。“科技的进步不可想象。”这是杨国强对于人类社会发展远景的整体判断。\
# 在他看来，只有紧跟时代浪潮的企业，才可以始终伫立于时代潮头。他表示：“现在机器人技术已经比较成熟，如果我们有足够优秀的人把这些做出来，我们会成为最先进的房地产公司，我们现在要朝着一个高科技企业去做，建筑是这样，物业管理也是。\
# 我也曾经在工地做过建筑工人，重复的高强度劳动说不过去。我们要迎接‘机器人建房子’的到来，只是时间问题，绝对要做出来，这是我们未来强大竞争力的源泉。\
# ”事实上，杨国强在更早之前便袒露过自己对机器人业务寄予的期望：“我梦想着建筑工人做的繁重、重复的劳动由机器人所替代……它首先是符合我们对零伤亡和安全的追求，第二能使我们的质量提升，第三能使我们的效率提升。\
# ”据悉，碧桂园将进一步投身智能制造，瞄准世界科技前沿，依托广东机器人谷，自主研发，打造现代机器人产业生态圈，助力国家科技进步。\
# 目前，碧桂园已经成立了专门从事机器人研究的博智林公司，并从全球范围引进了大量专业人才，紧锣密鼓开展着相关研发工作。\
# 杨国强也认为，科技让国家更强大，能帮助人们从繁重的工作中解放出来，让人们过上更好的生活。未来一些繁重、危险的工作将被机器人取代，“今天加大对科技的投入正是时候，如果再不努力的话就落后了。"
    text = '这件事情真是太恶劣了，居然发生这样的车祸！！！'
    if len(text) > 200:
        text = abstract(text)
    print(text)
    test_D,test_model_pred = data_prepare(text)
    res = predict(model,test_D,test_model_pred)
    print(res)