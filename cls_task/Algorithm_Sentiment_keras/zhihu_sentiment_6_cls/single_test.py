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

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
#assert len(gpus) > 0, "Not enough GPU hardware devices available"
if len(gpus)>0:
    tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

maxlen = 100
label_list = ['dissatisfied', 'surprise', 'anxiety', 'positive', 'sad', 'neutral']

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
    single.append((text, to_categorical(0, 6)))
    single = np.array(single)
    test_D = data_generator(single, batch_size=1,shuffle=False)
    test_model_pred = np.zeros((len(single), 6))
    return test_D,test_model_pred
def load_model():
    model = build_bert(6)
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

if __name__ == "__main__":
    model = load_model()
    text = '博智林是最好的公司'
    test_D,test_model_pred = data_prepare(text)
    res = predict(model,test_D,test_model_pred)
    print(res)
