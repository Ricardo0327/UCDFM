import jieba
import numpy as np
from gensim.models import Word2Vec  #word2vec模型其实就是简单化的神经网络
#这个模型是如何定义数据的输入和输出呢？一般分为CBOW(Continuous Bag-of-Words 与Skip-Gram两种模型。
# CBOW模型的训练输入是某一个特征词的上下文相关的词对应的词向量，而输出就是这特定的一个词的词向量。　
# Skip-Gram模型和CBOW的思路是反着来的，即输入是特定的一个词的词向量，而输出是特定词对应的上下文词向量。
# CBOW对小型数据库比较合适，而Skip-Gram在大型语料中表现更好。
from sklearn.decomposition import PCA
from copy import deepcopy
import random
import math
import spacy
import pandas as pd
from jieba import analyse
import operator
import pickle
import copy
from scipy.spatial.distance import cosine
from gensim.models import FastText
from collections import Counter
import sys
from sklearn.cluster import KMeans
import argparse
import time, os
import copy
from random import sample
import torchvision
from sklearn.model_selection import train_test_split#train_test_split返回切分的数据集train/test
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
import re
#下面的是加入的导入包
import networkx as nx
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder
import pandas as pd

# def structuralGraph():
#     dataReal = pd.read_csv("data/outputRealTh.csv")
#     dataFake = pd.read_csv("data/outputFakeTh.csv")
#     g = nx.Graph()
#
#     # gTrain = nx.Graph()
#     # gTest = nx.Graph()
#     gRealR = nx.Graph()
#     gRealF = nx.Graph()
#     gFakeR = nx.Graph()
#     gFakeF = nx.Graph()
#     # print(data.shape)#它的功能是查看矩阵或者数组的维数
#     # #查询数据的前五行
#     # print(data.head())
#     # df.loc[行标签，列标签]：
#     # 通过标签查询指定的数据，第一个值为行标签，第二值为列标签。
#     # 当第二个参数为空时，查询的是单个或多个行的所有列。
#     # 如果查询多个行、列的话，则两个参数用列表表示。
#     nodeLeft = dataReal.loc[:, ['id_1', 'id1_target']]
#     nodeRight = dataReal.loc[:, ['id_2', 'id2_target']]
#     # print('len(nodeLeft):',len(nodeLeft))
#     # or (
#     #            nodeLeft.iat[i, 1] == 'company' and nodeRight.iat[i, 1] == 'tvshow') or (
#     #            nodeLeft.iat[i, 1] == 'tvshow' and nodeRight.iat[i, 1] == 'company')
#     for i in range(1000):
#         if (nodeLeft.iat[i, 1] == 'government' and nodeRight.iat[i, 1] == 'government') or (
#                 nodeLeft.iat[i, 1] == 'government' and nodeRight.iat[i, 1] == 'tvshow') or (
#                 nodeLeft.iat[i, 1] == 'tvshow' and nodeRight.iat[i, 1] == 'government'):
#             gRealF.add_edge(nodeLeft.iat[i, 0], nodeRight.iat[i, 0])
#             g.add_edge(nodeLeft.iat[i, 0], nodeRight.iat[i, 0])
#         else:
#             gRealR.add_edge(nodeLeft.iat[i, 0], nodeRight.iat[i, 0])
#             g.add_edge(nodeLeft.iat[i, 0], nodeRight.iat[i, 0])
#
#     nodeLeftF = dataFake.loc[:, ['id_1', 'id1_target']]
#     nodeRightF = dataFake.loc[:, ['id_2', 'id2_target']]
#     # print('len(nodeLeftF):',len(nodeLeftF))
#     for i in range(800):#200 400 600 800
#         if (nodeLeftF.iat[i, 1] == 'government' and nodeRightF.iat[i, 1] == 'government') or (
#                 nodeLeftF.iat[i, 1] == 'government' and nodeRightF.iat[i, 1] == 'tvshow') or (
#                 nodeLeftF.iat[i, 1] == 'tvshow' and nodeRightF.iat[i, 1] == 'government') :
#             gFakeF.add_edge(nodeLeftF.iat[i, 0], nodeRightF.iat[i, 0])
#             g.add_edge(nodeLeftF.iat[i, 0], nodeRightF.iat[i, 0])
#         else:
#             gFakeR.add_edge(nodeLeftF.iat[i, 0], nodeRightF.iat[i, 0])
#             g.add_edge(nodeLeftF.iat[i, 0], nodeRightF.iat[i, 0])
#     # node2vec = Node2Vec(g, dimensions=128, walk_length=5, num_walks=10, workers=1)
#     # model = node2vec.fit(window=10, min_count=1, batch_words=1)
#     # edges_embs = HadamardEmbedder(keyed_vectors=model.wv)
#     # edges_kv = edges_embs.as_keyed_vectors()
#     # EDGES_EMBEDDING_FILENAME = 'trainFacebookPage.txt'
#     # edges_kv.save_word2vec_format(EDGES_EMBEDDING_FILENAME)
#     # node2vec = Node2Vec(g1, dimensions=128, walk_length=5, num_walks=10, workers=1)
#     # model = node2vec.fit(window=10, min_count=1, batch_words=1)
#     # edges_embs = HadamardEmbedder(keyed_vectors=model.wv)
#     # edges_kv = edges_embs.as_keyed_vectors()
#     # EDGES_EMBEDDING_FILENAME = 'testFacebookPage.txt'
#     # edges_kv.save_word2vec_format(EDGES_EMBEDDING_FILENAME)
#
#     node2vec = Node2Vec(g, dimensions=128, walk_length=5, num_walks=10, workers=1)
#     model = node2vec.fit(window=10, min_count=1, batch_words=1)
#     edges_embs = HadamardEmbedder(keyed_vectors=model.wv)
#     edges_kv = edges_embs.as_keyed_vectors()
#     EDGES_EMBEDDING_FILENAME = 'all17.txt'
#     edges_kv.save_word2vec_format(EDGES_EMBEDDING_FILENAME)
#
#     return g, gRealR, gRealF, gFakeR, gFakeF

#re 模块使 Python 语言拥有全部的正则表达式功能。
#Python 的 re 模块提供了re.sub用于替换字符串中的匹配项。
#用空格替代string里面的中文
def clean_str_sst(string):
    string = re.sub(r'[^\u4e00-\u9fa5A-Za-z0-9]', '', string)
    return string.strip().lower()


class News(object):  # self,contents,ids=None,author='',stopwords={},language='ch'

    def __init__(self, contents, label_str=None, ids=None, author='', stopwords={}, language='ch'):
        self.ids = ids
        self.contents = contents
        self.label_str = label_str
        self.author = author.strip().lower()
        self.language = language
        self.event_id = []
        self.clusterid = None
        self.label = None
        self.vector = None
        self.weight = None
        self.event_num = None
        self.author_news_num = None
        self.vote_value = None
        self.vote_sort = None
        self.analyzContents(stopwords)
        return

    def __str__(self):
        return self.contents

    def analyzContents(self, stopwords):  # 该函数的作用就是去掉停用,并且将分词之后得到的字符串用self.words存储,将分词之后并且去掉停用词的字符串用self.word_list存储
        words = [];
        words_split = [];
        real_words = []
        line_clean = self.clean_str_sst()
        if self.language == 'ch':
            words_split = list(jieba.cut(line_clean, cut_all=False))  # 将对应中文进行分词,words_split里面包含的是分词列表
        words.append(' '.join(words_split))  # 用空格将列表中的每一个词都分割开,返回的是一个字符串
        for word in words_split:  # 删除停用词
            if word not in stopwords:
                real_words.append(word)
        self.words = words  # self.words包含了所有的words
        self.word_list = real_words  # self.word_list只包含去掉停用词之后的所有words
        return 0

    def clean_str_sst(self):  # 这个函数是进行字符串预处理,但是仅限于英文
        string = self.contents
        string = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", string)  # 加上r防止字符串被转义,这一句话的意思是除去除了括号内的字符的所有字符
        return string.strip()


class Event:

    def __init__(self, news_list, vector, event_id, keyword, old):
        self.news_list = news_list
        self.vector = vector
        self.id = event_id
        self.keyword = [keyword]
        self.old = old
        self.represent = None
        self.label = None
        self.update = None
        self.confidence = None


def real_met(a, b):  # a是预测值,b是真实值
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(len(a)):
        if a[i] == 1 and b[i] == 1:
            TN = TN + 1
        if a[i] == 1 and b[i] == 0:
            FN = FN + 1
        if a[i] == 0 and b[i] == 1:
            FP = FP + 1
        if a[i] == 0 and b[i] == 0:
            TP = TP + 1
    return TP, FN, FP, TN


def fake_met(a, b):  # a是预测值,b是真实值
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(len(a)):
        if a[i] == 1 and b[i] == 1:
            TP = TP + 1
        if a[i] == 1 and b[i] == 0:
            FP = FP + 1
        if a[i] == 0 and b[i] == 1:
            FN = FN + 1
        if a[i] == 0 and b[i] == 0:
            TN = TN + 1
    return TP, FN, FP, TN


def cal_precision(a, b, c):
    if c == 0:
        TP, FN, FP, TN = fake_met(a, b)
    else:
        TP, FN, FP, TN = real_met(a, b)
    if TP + FP == 0:
        return 0
    else:
        return round(TP / (TP + FP), 4)


def cal_recall(a, b, c):
    if c == 0:
        TP, FN, FP, TN = fake_met(a, b)
    else:
        TP, FN, FP, TN = real_met(a, b)
    if TP + FN == 0:
        return 0
    else:
        return round(TP / (TP + FN), 4)


def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData / np.tile(ranges, (m, 1))
    return normData


def cal_f1(a, b, c):  # c=0表示是fake news为正样本,c=1表示real news为正样本
    precision = cal_precision(a, b, c)
    recall = cal_recall(a, b, c)
    if precision + recall == 0:
        return 0
    else:
        return round(2 * precision * recall / (precision + recall), 4)


def cal_auc(a, b):
    y_true = []
    y_pred = []
    for i in range(len(a)):
        y_pred.append(a[i])
        y_true.append(b[i])
    auc_score = metrics.roc_auc_score(y_true, y_pred)#cui加了# metrics.
    return round(auc_score, 4)


def cal_loss_func(a, b):
    count = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            count = count + 1
    result = round((float(count) / len(a)), 4)
    return result


class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj
    def read(self, size):
        return self.fileobj.read(size).decode('utf-8')
    def readline(self, size=-1):
        return self.fileobj.readline(size).decode('utf-8')

def pre():
    dataLabel = pd.read_csv("dataset/train_label.txt", sep=',', header=None)
    data = pd.read_csv("dataset/train.txt", sep=',', header=None)
    a = len(data.iloc[:, :])
    train_data = []
    test_data = []
    for i in range(a):
        news = News(" ")
        #print(re.sub("\D", "", nodeL[i][0]), re.sub("\D", "", nodeR[i][0]), edges[i][0], prevector[i])
        # vector = preVector[i]
        # label = random.randint(0, 1)
        # if G.has_edge(nodel,noder):#边在图中
        #     label = 1
        # else:
        #     label = 0
        label = dataLabel.iloc[i, 0]
        clusterid = random.randint(0, 15)
        # clusterid = random.randint(0, 20)
        vector = np.array(data.iloc[i, 0:32])
        news.vector = vector
        news.label = label
        news.clusterid = clusterid
        # n = [vector, label, clusterid]
        train_data.append(news)

    dataLabelTest = pd.read_csv("dataset/train_label.txt", sep=',', header=None)
    dataTest = pd.read_csv("dataset/train.txt", sep=',', header=None)
    a = len(dataTest.iloc[:, :])
    for i in range(a):
        news = News(" ")
        # print(re.sub("\D", "", nodeL[i][0]), re.sub("\D", "", nodeR[i][0]), edges[i][0], prevector[i])
        # vector = preVector[i]
        # label = random.randint(0, 1)
        # if G.has_edge(nodel,noder):#边在图中
        #     label = 1
        # else:
        #     label = 0
        label = dataLabelTest.iloc[i, 0]
        clusterid = random.randint(0, 15)
        # clusterid = random.randint(0, 20)
        vector = np.array(dataTest.iloc[i, 0:32])
        news.vector = vector
        news.label = label
        news.clusterid = clusterid
        # n = [vector, label, clusterid]
        test_data.append(news)

#从340到3899月26注释掉的
    # data = pd.read_csv("all17.txt", sep=' ', skiprows=1, header=None)
    # edges = np.array(data.iloc[:, 0:1]) + np.array(data.iloc[:, 1:2])
    # preVector = np.array(data.iloc[:, 2:])
    # nodeL = np.array(data.iloc[:, 0:1])
    # nodeR = np.array(data.iloc[:, 1:2])
    # train_data = []
    # test_data = []
    # countRealR, countRealF, countFakeR, countFakeF = 0, 0, 0, 0
    # for i in range(len(edges)):
    #     newsTrain = News(" ")
    #     newsTest = News(" ")
    #     nodel = int(re.sub("\D", "", nodeL[i][0]))
    #     noder = int(re.sub("\D", "", nodeR[i][0]))
    #     # print(re.sub("\D", "", nodeL[i][0]), re.sub("\D", "", nodeR[i][0]), edges[i][0], prevector[i])
    #     # vector = preVector[i]
    #     # label = random.randint(0, 1)
    #     # if label == 0:
    #     #     label = random.randint(0, 1)
    #     # GRealR, GRealF, GFakeR, GFakeF
    #     if GRealR.has_edge(nodel, noder) or GFakeR.has_edge(nodel, noder): #训练集 # 边在图中
    #         if GRealR.has_edge(nodel, noder):
    #             label = 1
    #             countRealR = countRealR + 1
    #         else:
    #             label = 0
    #             countFakeR = countFakeR + 1
    #         clusterid = random.randint(0, 15)
    #         newsTrain.vector = preVector[i]
    #         newsTrain.label = label
    #         newsTrain.clusterid = clusterid
    #         train_data.append(newsTrain)
    #     elif GRealF.has_edge(nodel, noder) or GFakeF.has_edge(nodel, noder):#测试集
    #         if GRealF.has_edge(nodel, noder):
    #             label = 1
    #             countRealF = countRealF + 1
    #         else:
    #             label = 0
    #             countFakeF = countFakeF + 1
    #         clusterid = random.randint(0, 15)
    #         newsTest.vector = preVector[i]
    #         newsTest.label = label
    #         newsTest.clusterid = clusterid
    #         test_data.append(newsTest)
    #     else :
    #         continue
        # clusterid = random.randint(0, 15)
        # news.vector = preVector[i]
        # news.label = label
        # news.clusterid = clusterid
        # test_data.append(news)
    # print('countRealR:',countRealR, 'countRealF:', countRealF, 'countFakeR:',countFakeR,'countFakeF:',countFakeF)
    return train_data, test_data

#n.Module 是所有神经网络单元（neural network modules）的基类
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
        self.class_classifier.add_module('c_fc1', nn.Linear(32, 16))
        # self.class_classifier.add_module('c_fc1', nn.Linear(self.hidden_size, 30))
        self.class_classifier.add_module('c_fc1_relu', nn.ReLU())
        self.class_classifier.add_module('c_fc2', nn.Linear(16, 16))
        # self.class_classifier.add_module('c_fc2', nn.Linear(30, 10))
        self.class_classifier.add_module('c_fc2_relu', nn.ReLU())
        self.class_classifier.add_module('c_fc3', nn.Linear(16, 16))
        # self.class_classifier.add_module('c_fc3', nn.Linear(10, 2))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))  # 对每一行进行softmax

        ###Event Classifier让eann学习共性
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(32, 16))
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

def to_var(x):  # 使用cuda计算x
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_np(x):  # 这里的x是gpu的tensor,这个函数就是将tensor转化成numpy
    return x.data.cpu().numpy()  # gpu的tensor不能直接转换成numpy,需要先转换成cpu的tensor再转换成numpy，取值操作


def main(batch_size, num_epochs, learning_rate, output_file, hidden_dim):
    train_data, test = pre()#预处理
    print(train_data, test)
    print(type(train_data), type(test))
    train, validate = train_test_split(train_data, test_size=0.7)#train_test_split返回切分的数据集train/test
    train_dataset = []
    validate_dataset = []
    test_dataset = []
    for i in train:
        print('len(i.vector):', len(i.vector))
        vectors = torch.tensor(i.vector, dtype=torch.float32)
        label = torch.tensor(i.label, dtype=torch.float32)
        clusterid = torch.tensor(i.clusterid, dtype=torch.float32)
        # print('len--vector:', len(vectors))
        # vectors = torch.tensor(i[0], dtype=torch.float32)
        # label = torch.tensor(i[1], dtype=torch.float32)
        # clusterid = torch.tensor(i[2], dtype=torch.float32)
        m = [vectors, label, clusterid]
        train_dataset.append(m)  # train_dataset中的每个元素都是一个列表,列表中含有三个元素分别对应:向量,标签,所在cluster的id
    for i in validate:
        vectors = torch.tensor(i.vector, dtype=torch.float32)
        label = torch.tensor(i.label, dtype=torch.float32)
        clusterid = torch.tensor(i.clusterid, dtype=torch.float32)
        # vectors = torch.tensor(i[0], dtype=torch.float32)
        # label = torch.tensor(i[1], dtype=torch.float32)
        # clusterid = torch.tensor(i[2], dtype=torch.float32)
        m = [vectors, label, clusterid]
        validate_dataset.append(m)
    for i in test:
        # if i.label == 1:
        #     print('test-i.label:',1)
        vectors = torch.tensor(i.vector, dtype=torch.float32)
        label = torch.tensor(i.label, dtype=torch.float32)
        clusterid = torch.tensor(i.clusterid, dtype=torch.float32)
        # vectors = torch.tensor(i[0], dtype=torch.float32)
        # label = torch.tensor(i[1], dtype=torch.float32)
        # clusterid = torch.tensor(i[2], dtype=torch.float32)
        m = [vectors, label, clusterid]
        test_dataset.append(m)
    # print(train_dataset)
    print('train length', len(train_dataset))
    print('validate length', len(validate_dataset))
    print('test length', len(test_dataset))
    #将整个数据集按7:1:2的比例分割为训练、验证和测试集
    # 该接口的目的：将自定义的Dataset根据batch size大小、是否shuffle等封装成一个Batch Size大小的Tensor，用于后面的训练
    # dataset(Dataset): 传入的数据集
    # batch_size(int, optional): 每个batch有多少个样本
    # shuffle(bool, optional): 在每个epoch开始的时候，对数据进行重新排序
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)  # train_loader中的每一个数据都是32*10维度的数据,其中batchsize是32,embedding是10
    validate_loader = DataLoader(dataset=validate_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)# 创建一个 DataLoader 对象，
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    print('building model')
    model = CNN_Fusion(hidden_dim)#hidden_dim=10
    if torch.cuda.is_available():#显卡驱动
        print("CUDA")
        model.cuda()
    criterion = nn.CrossEntropyLoss()  # 创建一个交叉熵对象
    optimizer = torch.optim.Adam(list(model.parameters()),
                                 lr=learning_rate)  # model.parameters()是需要优化的参数,也就是说这个模型的中的所有参数都是需要被优化的
    #params (iterable) – 待优化参数的iterable或者是定义了参数组的dict
    #lr (float, 可选) – 学习率（默认：1e-3）
    #lr：同样也称为学习率或步长因子，它控制了权重的更新比率（如 0.001）。
    # 较大的值（如 0.3）在学习率更新前会有更快的初始学习，而较小的值（如 1.0E-5）会令训练收敛到更好的性能。
    iter_per_epoch = len(train_loader)
    print('len of train_loader', iter_per_epoch)#num is 8
    best_validate_acc = 0.000
    best_test_acc = 0.000
    best_loss = 100
    best_validate_dir = ''
    best_list = [0, 0]
    print('training model')
    adversarial = True
    # Train the Model
    #Epoch 使用训练集的全部数据对模型进行一次完整训练，被称之为“一代训练”
    #Batch 使用训练集中的一小部分样本对模型权重进行一次反向传播的参数更新，这一小部分样本被称为“一批数据”
    #Iteration 使用一个Batch数据对模型进行一次参数更新的过程，被称之为“一次训练”
    #Epoch（时期）：
    #当一个完整的数据集通过了神经网络一次并且返回了一次，这个过程称为一次>epoch。（也就是说，所有训练样本在神经网络中都 进行了一次正向传播 和一次反向传播 ）
    #再通俗一点，一个Epoch就是将所有训练样本训练一次的过程。
    for epoch in range(num_epochs):#num_epochs is 50
        p = float(epoch) / 100
        lr = 0.001 / (1. + 10 * p) ** 0.75
        optimizer.lr = lr
        start_time = time.time()
        cost_vector = []
        class_cost_vector = []
        domain_cost_vector = []
        acc_vector = []
        valid_acc_vector = []
        test_acc_vector = []
        vali_cost_vector = []
        test_cost_vector = []
        for i, (train_data, train_labels, event_labels) in enumerate(
                train_loader):  # 每一次迭代都要更新参数,batchsize是每一次迭代所训练的数据量,经过一次迭代之后模型的参数更新
            optimizer.zero_grad()  # 在每一轮batch的时候都要初始化,将梯度设置为
            train_data = to_var(train_data)
            train_labels = to_var(train_labels),
            event_labels = to_var(event_labels)
            # print('train i:', i)
            # # print('train  train_data:', train_data)
            # print('train  train_labels:', train_labels)
            # print('train  event_labels:', event_labels)
            class_outputs, domain_outputs = model(train_data)  # 每一轮的batch_size有32个
            # print('typeclass_outputs:', type(class_outputs), 'typedomain_outputs:', type(domain_outputs))
            # print('class_outputs:', class_outputs, 'domain_outputs:', domain_outputs)
            train_labels = train_labels[0]
            train_labels = train_labels.long()
            event_labels = event_labels.long()
            # class loss
            class_loss = criterion(class_outputs,
                                   train_labels)  # criterion是交叉熵损失函数,这里也就是分别计算class_output和train_labels之间的交叉熵,就是detector的loss函数

            # Event Loss
            domain_loss = criterion(domain_outputs, event_labels)
            # 定义loss
            loss = class_loss - domain_loss
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

            _, argmax = torch.max(class_outputs, 1)
            cross_entropy = True

            # 定义accuracy
            if True:
                accuracy = (train_labels == argmax.squeeze()).float().mean()
            else:
                _, labels = torch.max(train_labels, 1)
                accuracy = (labels.squeeze() == argmax.squeeze()).float().mean()
            # class 部分的loss值
            class_cost_vector.append(class_loss.item())

            # domain部分的loss值
            domain_cost_vector.append(domain_loss.item())

            # 总的loss的值
            cost_vector.append(loss.item())

            # 总的accuracy
            acc_vector.append(accuracy.item())
        # 开始验证
        model.eval()  # 不开启batchnomalization和dropout
        validate_acc_vector_temp = []
        for i, (validate_data, validate_labels, event_labels) in enumerate(validate_loader):
            validate_data = to_var(validate_data)
            validate_labels = to_var(validate_labels)
            event_labels = to_var(event_labels)
            validate_outputs, domain_outputs = model(validate_data)
            _, validate_argmax = torch.max(validate_outputs, 1)
            #input是softmax函数输出的一个tensor
            #dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值
            validate_labels = validate_labels.long()#转化为长整形
            vali_loss = criterion(validate_outputs, validate_labels)
            validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()#去掉维数为1的的维度
            vali_cost_vector.append(vali_loss.item())#item()方法把字典中每对key和value组成一个元组，并把这些元组放在列表中返回。
            validate_acc_vector_temp.append(validate_accuracy.item())
        validate_acc = np.mean(validate_acc_vector_temp)#求取均值
        valid_acc_vector.append(validate_acc)#每组均值添加到向量组
        model.train()
        print('Epoch [%d/%d],  Loss: %.4f, Class Loss: %.4f, domain loss: %.4f, Train_Acc: %.4f,  Validate_Acc: %.4f.'
              % (
                  epoch + 1, num_epochs, np.mean(cost_vector), np.mean(class_cost_vector), np.mean(domain_cost_vector),
                  np.mean(acc_vector), validate_acc))

        if validate_acc > best_validate_acc:#取最好的精确度
            best_validate_acc = validate_acc
            if not os.path.exists(output_file):#os.path.exists()就是判断括号里的文件是否存在的意思，括号内的可以是文件路径。
                os.mkdir(output_file)#创建文件目录

            best_validate_dir = output_file + str(epoch + 1) + '.pkl'
            torch.save(model.state_dict(), best_validate_dir)#其中dir表示保存文件的绝对路径+保存文件名，如'/home/qinying/Desktop/modelpara.pth'
            #torch.nn.Module模块中的state_dict变量存放训练过程中需要学习的权重和偏执系数，
            # state_dict作为python的字典对象将每一层的参数映射成tensor张量，
            # 需要注意的是torch.nn.Module模块中的state_dict只包含卷积层和全连接层的参数，
            # 当网络中存在batchnorm时，例如vgg网络结构，torch.nn.Module模块中的state_dict也会存放batchnorm's running_mean

        duration = time.time() - start_time

    # Test the Model
    print('testing model')
    model = CNN_Fusion(hidden_dim)
    model.load_state_dict(torch.load(best_validate_dir))#然后加载模型时一般用
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    test_score = []
    test_pred = []
    test_true = []
    for i, (test_data, test_labels, event_labels) in enumerate(test_loader):
        test_data = to_var(test_data)
        test_labels = to_var(test_labels)
        event_labels = to_var(event_labels)
        test_outputs, domain_outputs = model(test_data)

        _, test_argmax = torch.max(test_outputs, 1)#dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值
        if i == 0:
            test_score = to_np(test_outputs)#取值操作
            test_pred = to_np(test_argmax)
            test_true = to_np(test_labels)
        else:
            test_score = np.concatenate((test_score, to_np(test_outputs)), axis=0)#按轴axis连接array组成一个新的array，axis=0表示按行连接
            test_pred = np.concatenate((test_pred, to_np(test_argmax)), axis=0)
            test_true = np.concatenate((test_true, to_np(test_labels)), axis=0)
            # print('test_score:',test_score, 'test_pred:', test_pred, 'test_true:', test_true)

    # print('1')
    test_accuracy = metrics.accuracy_score(test_true, test_pred)
    #分类准确率分数是指所有分类正确的百分比。分类准确率这一衡量分类器的标准比较容易理解，但是它不能告诉你响应值的潜在分布，并且它也不能告诉你分类器犯错的类型。
    # print('2')
    fake_pre = cal_precision(test_pred, test_true, 0)
    # print('3')
    fake_recall = cal_recall(test_pred, test_true, 0)
    # print('4')
    fake_f1 = cal_f1(test_pred, test_true, 0)
    # print('5')
    real_pre = cal_precision(test_pred, test_true, 1)
    real_recall = cal_recall(test_pred, test_true, 1)
    real_f1 = cal_f1(test_pred, test_true, 1)
    test_score_convert = [x[1] for x in test_score]
    test_aucroc = metrics.roc_auc_score(test_true, test_score_convert, average='macro')
    test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)
    #Confusion matrix
    #混淆矩阵是由false positives，falsenegatives，true positives和true negatives组成的两行两列的表格。
    # 它允许我们做出更多的分析，而不仅仅是局限在准确率。
    # 准确率对于分类器的性能分析来说，并不是一个很好地衡量指标，因为如果数据集不平衡（每一类的数据样本数量相差太大），很可能会出现误导性的结果。

    print('accuracy :', test_accuracy)
    print('auc :', test_aucroc)
    print('fake precision :', fake_pre)
    print('fake recall :', fake_recall)
    print('fake f1 :', fake_f1)
    print('real precision :', real_pre)
    print('real recall :', real_recall)
    print('real f1 :', real_f1)


if __name__ == '__main__':  # batch_size,num_epochs,learning_rate,output_file,hidden_dim
    output_file = 'output/output.txt'
    # G, GRealR, GRealF, GFakeR, GFakeF = structuralGraph()
    main(32, 50, 0.005, output_file, 16)
