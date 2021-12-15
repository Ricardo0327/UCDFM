from sklearn.linear_model import LogisticRegression
import numpy as np
import sklearn.svm as svm
from sklearn.metrics import accuracy_score
from filterpy.kalman import KalmanFilter
import math
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate,LSTM,Average,Lambda
from keras.models import Sequential#按顺序建层
from keras.layers import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
import keras
from keras import backend as K
from keras import optimizers
from sklearn.preprocessing import StandardScaler
from collections import Counter #引入Counter
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import keras.backend as K
import keras.layers as KL
import keras.models as KM
from keras import models
from keras import layers
from keras.models import Model
import torch.nn as nn
from keras import backend as K
from keras.layers import Embedding, Input, Dense, Conv1D, MaxPooling1D, Dense, Flatten, Lambda, LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
import sklearn.svm as svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
def build_model():
    model = Sequential()
    model.add(Dense(64, input_dim=32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model
def l_average(x):
    return K.mean(x,axis=1)

def lstm(x_train, y_train, x_test, y_test):  # input要是list of input,也就是每个input是张量,但是训练中用于输入的是一个列表,列表中的元素是张
    model = Sequential()
    # 构建embedding层。128代表了embedding层的向量维度
    model.add(Embedding(5000, 32))
    # 构建LSTM层
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    # 构建最后的全连接层，注意上面构建LSTM层时只会得到最后一个节点的输出， 如果需要输出每个时间点的结果，需要将return_sequences=True
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=500, epochs=15, validation_data=(x_test, y_test))

    score = model.evaluate(x_train, y_train, batch_size=500)
    print('test loss:', score[0])
    print('test accuracy:', score[1])
def Model1(x_train, y_train):
    model = Sequential()
    model.add(Dense(input_dim=32, units=50, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dense(units=25, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dense(units=10, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dense(units=1, activation='softmax', kernel_initializer='random_uniform'))
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)#lr学习率
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy'])#loss损失函数binary_crossentropy适用于二分类，optimizer优化器
    model.fit(x_train, y_train, batch_size=500, epochs=200, verbose=0)
    return model
news_user_matrix = np.loadtxt('news_user_matrix_1.txt')

def model2(x_train, y_train):
    embedding_dim = 50

    model = Sequential()
    model.add(layers.Embedding(input_dim=32,
                               output_dim=2,
                               input_length=32))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=50, epochs=20, verbose=0)
    model.summary()
def model(x_train, y_train):
        model = Sequential()
        model.add(Dense(input_dim=32, units=50, activation='relu', kernel_initializer='random_uniform'))
        model.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
        model.add(Dense(units=1, activation='softmax', kernel_initializer='random_uniform'))
        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy'])
        model.fit(x_train, y_train, batch_size=500, epochs=20, verbose=0)
        return model

class CNN_Fusion(nn.Module):  # 定义网络的时候需要继承nn.module并且实现其forward方法
    def __init__(self, hidden_dim):  # 定义模型结构
        super(CNN_Fusion, self).__init__()#调用父类的构造函数
        self.hidden_size = hidden_dim
        ## Class  Classifier
        self.class_classifier = nn.Sequential()
        #nn.Sequential()一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。
        #add_module,添加到分类器中，第一个参数是各层的名字，

        #class torch.nn.Linear（in_features，out_features，bias = True ）[来源]
        #对传入数据应用线性变换：y = A x+ b
        # print(self.hidden_size)
        self.class_classifier.add_module('c_fc1', nn.Linear(32, 1684))
        # self.class_classifier.add_module('c_fc1', nn.Linear(self.hidden_size, 30))
        self.class_classifier.add_module('c_fc1_relu', nn.ReLU())
        self.class_classifier.add_module('c_fc2', nn.Linear(16, 16))
        # self.class_classifier.add_module('c_fc2', nn.Linear(30, 10))
        self.class_classifier.add_module('c_fc2_relu', nn.ReLU())
        self.class_classifier.add_module('c_fc3', nn.Linear(16, 16))
        # self.class_classifier.add_module('c_fc3', nn.Linear(10, 2))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))  # 对每一行进行softmax

        ###Event Classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(128, 16))
        # self.domain_classifier.add_module('d_fc1', nn.Linear(self.hidden_size, 30))
        self.domain_classifier.add_module('relu_f1', nn.ReLU())
        self.domain_classifier.add_module('d_fc2', nn.Linear(16, 16))
        # self.domain_classifier.add_module('d_fc2', nn.Linear(30, 50))
        self.domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))#这里的dim=0其实就是张量的0轴，dim=1就是张量的1轴。，对每一行进行softmax
        #nn.Softmax()计算出来的值，其和为1，也就是输出的是概率分布

    def forward(self, x):#自定义向前计算的函数
        # print('x-------------------:', x)
        class_output = self.class_classifier(x)  # 分类器中的参数就是输入分类器的向量,events就是输入
        domain_output = self.domain_classifier(
            x)  # 这句话的意思就是reverse_feature这个特征向量作为输入,输出是domain_output,这个也是一个向量,其中的每一个值都是被分到该event的
        return class_output, domain_output
#Squential将网络层和激活函数结合起来，输出激活后的网络节点。
#
# user_num = 100
# news_num = 10000
news_user_matrix = np.loadtxt('news_user_matrix_1.txt')

# where user_news_matrix_i_j=1 means the user j posts the news i
label = np.loadtxt('label_1.txt')
print(np.where(label == 1)[0].shape)
print(np.where(label == 0))
print(np.where(label == -1)[0].shape)
user_num = news_user_matrix.shape[1]
news_num = news_user_matrix.shape[0]
index = np.arange(news_num)
np.random.seed(0)
np.random.shuffle(index)
unlabeled_index = index[: int(news_num * 0.8)]
X2_label = label[unlabeled_index]
label[unlabeled_index] = 0
labeled_index = np.where(np.abs(label) == 1)[0]
print(unlabeled_index.shape)
print(labeled_index.shape)

# where label=1 means truth, label=-1 means rumour, label=0 means unlabeled
news_vector = np.loadtxt('sentence2Vec_1.txt')
# pre-trained news embedding vectors
X1 = news_vector[labeled_index]
print(X1.shape)
Y1 = label[labeled_index]
print(Y1.shape)
print(Y1.sum())
X2 = news_vector[unlabeled_index]
print(np.where(X2_label == 1)[0].shape)
print(np.where(X2_label == -1)[0].shape)
Y1[np.where(Y1==-1)]=0
X2_label[np.where(X2_label==-1)]=0
test_accuracy = []
test_f1 = []
test_precision = []
test_recall = []
test_aucroc = []
'''clf = svm.SVC(C=10, kernel='rbf', degree=0.1, gamma=10, coef0=0.0,
              shrinking=True, probability=True, tol=0.001, cache_size=200,
              class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
              random_state=None)'''
#clf = LogisticRegression(solver='lbfgs')
clf = RandomForestClassifier()
clf.fit(X1, Y1)
Y2_pre = clf.predict(X2)
accuracy = metrics.accuracy_score(X2_label, Y2_pre)
f1 = metrics.f1_score( X2_label, Y2_pre)
precision = metrics.precision_score(X2_label, Y2_pre)
recall = metrics.recall_score(X2_label, Y2_pre)
aucroc = metrics.roc_auc_score(X2_label, Y2_pre)
test_accuracy.append(accuracy)
test_f1.append(f1)
test_precision.append(precision)
test_recall.append(recall)
test_aucroc.append(aucroc)
test_accuracy = np.mat(test_accuracy)
test_f1 = np.mat(test_f1)
test_precision = np.mat(test_precision)
test_recall = np.mat(test_recall)
test_aucroc = np.mat(test_aucroc)
np.savetxt("test_accuracy3.txt",test_accuracy,fmt="%f", delimiter=",")
np.savetxt("test_f13.txt", test_f1, fmt="%f", delimiter=",")
np.savetxt("test_precision3.txt",  test_precision, fmt="%f", delimiter=",")
np.savetxt("test_recall3.txt",test_recall, fmt="%f", delimiter=",")
np.savetxt("test_aucroc3.txt", test_aucroc, fmt="%f", delimiter=",")
#l=self_train_SVM(X1, Y1, X2, label, labeled_index, unlabeled_index, news_user_matrix, X2_label, C=1, kernel='linear', max_epoch=100)
#l=self_train_SVM(X1, Y1, X2, label, labeled_index, unlabeled_index, news_user_matrix, X2_label, C=1, kernel='linear', max_epoch=100)

#l=self_train_SVM(X1, Y1, X2, label, labeled_index, unlabeled_index, news_user_matrix, X2_label, C=1, kernel='linear', max_epoch=100)
#self_train_SVM(X1, Y1, X2, Y2, label, labeled_index, unlabeled_index, news_user_matrix, X2_label, C=1, kernel='linear', max_epoch=100)

#lstm(X1,Y1,X2, X2_label)

