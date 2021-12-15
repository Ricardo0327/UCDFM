import numpy as np
import sklearn.svm as svm
from sklearn.metrics import accuracy_score
from filterpy.kalman import KalmanFilter
import math
import re
import os
import jieba
import numpy as np
from gensim.models import Word2Vec#cbow是给出一组单词求出中心词,skip-gram是给出中心词求出上下文的词语,gensim是一个nlp库
from sklearn.decomposition import PCA
from copy import deepcopy
import random
import math
import spacy
import pandas as pd
from jieba import analyse
import operator
import scipy
from scipy.spatial.distance import cosine
from gensim.models import FastText
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras import backend as K
from keras import optimizers
import keras
from sklearn.metrics import roc_auc_score
from keras.utils import multi_gpu_model
def alph_filter(bel1 = 0.5 , bel2 = 0.6 , K = 10):
    """
           基于初值：X1、X2的置信度，计算K轮迭代的权重值
           Parameters
           ----------
           bel1 ： float
           X1的置信度

           bel2 ： float
           X2的置信度

           K： int
           迭代的次数

           Returns
           -------

           alph_new_list : list
           每次迭代的权重值

           Examples
           --------

        .. code-block:: Python

           import  Alph_filter

           result = Alph_filter.alph_filter(0.1,0.8,5)

           print(result)
    """

    noise = np.random.normal(0,0.1,K)

    my_filter = KalmanFilter(dim_x=2,dim_z=1)

    my_filter.x = np.array([[bel1],[bel2]]) #x1,x2的置信度，手动设置
    my_filter.P = np.array([[1.,0.],[0.,1.]]) #状态空间的协方差矩阵，表示Xn和Xn+1的相关性，为了简便设初值无关，但是在迭代过程中会自更新，所以不用太担心
    my_filter.F = np.array([[0.,1.],[1-(abs(bel2-bel1)/(bel1+bel2)),(bel2-bel1)/(bel1+bel2)]]) #状态转移矩阵，第一次的状态转移矩阵要根据公式和初值设置好，后面会更新
    my_filter.Q = np.array([[0.0001,0.],[0.,0.0001]]) #状态转移矩阵的协方差矩阵（Rt），描述外界噪声
    my_filter.H = np.array([[-1/(bel1+bel2),1/(bel1+bel2)]]) #观察矩阵
    my_filter.R = 0.1 #设置测量噪声为0.1

    alph_new_list = []

    for k in range(K):
        my_filter.predict()#预测
        x = my_filter.x
        Xn = float(x[0])
        Xn1 = float(x[1])
        if Xn1>Xn and Xn1-Xn > 0.0001 :
            alph = (Xn1-Xn)/(Xn1+Xn)
        elif Xn1<Xn and Xn-Xn1 > 0.0001 :
            alph = (Xn-Xn1)/(Xn1+Xn)
        else:
            pass
        #因为（Xn1-Xn)/(Xn1+Xn）这个估计量的性质不是很优秀，当Xn1-Xn过小时会发生溢出，所以若差值过小则不做更新，而且alph的这种估计先天偏好Xn（就是当前状态的前一个状态），如果能想到更好的估计量，可以更改计算公式
        my_filter.F = np.array([[0., 1.], [alph, 1-alph]])  # 状态转移矩阵更新
        my_filter.update(alph + noise[k])#更新，将alph的预测值+一个标准正态的随机变量作为测量值
        my_filter.H = np.array([[-1 / (Xn + Xn1), 1 / (Xn + Xn1)]])  # 观察矩阵更新
        x = my_filter.x
        Xn = float(x[0])
        Xn1 = float(x[1])
        if Xn1 > Xn and Xn1 - Xn > 0.001:
            alph = abs(Xn1 - Xn) / (Xn1 + Xn)
        elif Xn1 < Xn and Xn - Xn1 > 0.001:
            alph = abs(Xn - Xn1) / (Xn1 + Xn)
        else:
            pass
        alph_new_list.append(alph)

    return alph_new_list

def get_thre(Y3_real,Y3,thre,n):
    print("第",n,"次")
    l=1684
    print("第",n,"次进入函数里面的阈值",thre)
    po_index = np.where(Y3 >= thre)
    ne_index = np.where(Y3 < thre)
    Y3[po_index] = 1
    Y3[ne_index] = -1
    Y3_pre=np.array(Y3,copy=True)
    Y3_pre[np.where(Y3_pre == -1)] = 0
    Y3_real[np.where(Y3_real==-1)] = 0
    TP = np.dot(Y3_pre, Y3_real.T)
    print(TP)
    FP = np.sum(Y3_pre) - TP
    print(FP)
    FN = np.sum(Y3_real) - TP
    print(FN)
    TN = len(Y3_real) - TP - FP - FN
    print(TN)
    thre=(TP + TN) / (len(Y3_real))
    print("第", n, "次出去函数里面的阈值", thre)
    return thre


def self_train_SVM(X1, Y1, X2, label, labeled_index, unlabeled_index, news_user_matrix, X2_label, C=1, kernel='linear', max_epoch=200, t_method=None):
    l = X1.shape[0]
    u = X2.shape[0]
    N = l + u
    sample_train = int(0.05 * u)

    news_user_label_mat = np.multiply(label, news_user_matrix.T).T
    user_news_count = news_user_matrix.sum(0)
    user_truth_news_count = (np.abs(news_user_label_mat).sum(0) + news_user_label_mat.sum(0)) / 2
    unknown_any_label_index = np.where(np.abs(news_user_label_mat).sum(0) == 0)
    print("user_truth_news_count",user_truth_news_count)
    Wc = user_truth_news_count / user_news_count
    Wc[unknown_any_label_index] = 0.5
    x = Wc
    print("Wc", Wc)
    acc_iter_list = []

    '''clf = svm.SVC(C=10, kernel='rbf', degree=0.1, gamma=10, coef0=0.0,
                  shrinking=True, probability=True, tol=0.001, cache_size=200,
                  class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
                  random_state=None)'''
    maxlen = 60
    embed_size = 10
    tensor_input = Input(shape=(maxlen, embed_size))
    convs = []
    filter_sizes = [2, 3, 4, 5]
    for fsz in filter_sizes:
        l_conv = Conv1D(filters=100, kernel_size=fsz, activation='relu')(
            tensor_input)  # kernel_size中只用说明窗口大小就行,不用说宽度是多少,因为宽度默认是embedding size
        l_pool = MaxPooling1D(maxlen - fsz + 1)(l_conv)  # 此时是100个1
        l_pool = Flatten()(l_pool)  # 将数据压成1维的数据,也就是将数据变成1x100的形式
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)
    out = Dropout(0.3)(merge)
    output = Dense(50, activation='relu')(out)
    output = Dense(10, activation='relu')(output)
    output = Dense(units=2, activation='softmax')(output)
    clf = Model(tensor_input, output)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # model=multi_gpu_model(model, gpus=2)
    clf.compile(loss="binary_crossentropy", optimizer=adam, metrics=['binary_accuracy'])
    clf.fit(X1, Y1, batch_size=30 * 2, epochs=20)


    Y2 = clf.predict_proba(X2)[:, 1]
    sort_index = np.argsort(Y2)
    ne_index_train = sort_index[:sample_train]
    po_index_train = sort_index[-sample_train:]
    Y1_pro = clf.predict_proba(X1)[:, 1]
    # Y1_pre = clf.predict(X1)
    thre = 0
    # if t_method == 'Y1+w+highestACC_without_normal':
    label_c = np.array(label, copy=True)
    label_c[unlabeled_index] = Y2
    label_c[labeled_index] = Y1_pro
    zero_index = np.where(news_user_matrix == 0)
    label_v = (np.multiply(news_user_matrix, Wc).T + label_c).T
    label_v[zero_index] = 0
    label_v = label_v.T
    label_v = label_v.sum(0)
    Y2 = label_v[unlabeled_index]
    Y1_pro = label_v[labeled_index]
    Y1_real = np.array(Y1, copy=True)
    Y1_real[np.where(Y1 == -1)] = 0
    print()
    Y4 = label_v[labeled_index]
    Y3_real = np.array(Y1, copy=True)
    thre = get_thre(Y3_real, Y4, 0.74,0)
    #thre=1

    print("thre",thre)
    Y4[np.where(Y4 >= thre)] = 1
    Y4[np.where(Y4 < thre)] = -1
    po_index = np.where(Y2 >= thre)
    ne_index = np.where(Y2 < thre)
    Y2[po_index] = 1
    Y2[ne_index] = -1

    Y2_pre = np.array(Y2, copy=True)
    Y2_pre[np.where(Y2_pre == -1)] = 0
    Y2_real = np.array(X2_label, copy=True)
    Y2_real[np.where(Y2_real == -1)] = 0

    TP = np.dot(Y2_pre, Y2_real)
    FP = np.sum(Y2_pre) - TP
    FN = Y2_real.sum() - TP
    TN = u - TP - FP - FN
    acc_iter_list.append((TP + TN) / u)
    print(TP)
    print(TN)
    print(np.where(Y2_pre == 0)[0].shape)
    print(np.where(Y2_pre == 1)[0].shape)
    print((TP + TN) / u)
    # print(np.dot(Y2_pre, Y2_real))
    # print(np.sum(Y2_pre))
    # print(np.dot(Y2_pre, Y2_real) / np.sum(Y2_pre))

    # Y2 = np.expand_dims(Y2, 1)
    # X2_id = np.arange(u)
    label[unlabeled_index] = Y2

    news_user_label_mat = np.multiply(label, news_user_matrix.T).T
    user_news_count = news_user_matrix.sum(0)
    user_truth_news_count = (np.abs(news_user_label_mat).sum(0) + news_user_label_mat.sum(0)) / 2
    print("user_truth_news_count",user_truth_news_count)
    Wc = user_truth_news_count / user_news_count
    y = Wc
    P = 0.02
    Q = 0.008
    F = 1.0
    R = 0.01
    Wc, P, Q, F, R = kalman(x,y , P, Q, F, R)
    #rint("Wc1",Wc1)
    #print("Wc",Wc.shape)
    #print("Wc1", Wc1.shape)
    #Wc2 = alph_filter(Wc1, Wc)
    #print("Wc2",Wc2)c
    X_add = np.vstack([X2[po_index_train], X2[ne_index_train]])
    sample_train_po_label = np.ones(sample_train)
    y_add = np.append(sample_train_po_label, -sample_train_po_label)
    X3 = np.vstack([X1, X_add])
    Y3 = np.append(Y1, y_add)
    # X3 = np.vstack([X1, X2])
    # Y3 = np.append(Y1, Y2)
    n=0

    for i in range(max_epoch):
        clf.fit(X3, Y3)

        # Y2_pre = clf.predict(X2)
        # Y2_pre[np.where(Y2_pre == -1)] = 0
        # Y2_real = np.array(X2_label, copy=True)
        # Y2_real[np.where(Y2_real == -1)] = 0
        # prec = precision_score(Y2_real, Y2_pre)
        # print(prec)
        # precision_iter_list.append(prec)

        Y2 = clf.predict_proba(X2)[:, 1]
        sort_index = np.argsort(Y2)
        ne_index_train = sort_index[:sample_train]
        po_index_train = sort_index[-sample_train:]
        Y1_pro = clf.predict_proba(X1)[:, 1]
        #thre = 0
        '''if t_method == 'Y1+w+highestACC_without_normal':
            label_c = np.array(label, copy=True)
            label_c[unlabeled_index] = Y2
            label_c[labeled_index] = Y1_pro
            zero_index = np.where(news_user_matrix == 0)
            label_v = (np.multiply(news_user_matrix, Wc).T + label_c).T
            label_v[zero_index] = 0
            label_v = label_v.T
            label_v = label_v.sum(0)
            Y2 = label_v[unlabeled_index]
            Y1_pro = label_v[labeled_index]

            Y1_real = np.array(Y1, copy=True)
            Y1_real[np.where(Y1 == -1)] = 0
            thre = get_thre_by_acc(np.mat(Y1_real), np.mat(Y1_pro))
            print()
            print(thre)
            po_index = np.where(Y2 >= thre)
            ne_index = np.where(Y2 < thre)
            Y2[po_index] = 1
            Y2[ne_index] = -1'''
        #t_method == 'Y1+highestACC_with_max_min_normal':
        label_c = np.array(label, copy=True)
        label_c[unlabeled_index] = Y2
        zero_index = np.where(news_user_matrix == 0)
        label_v = (np.multiply(news_user_matrix, Wc).T + label_c).T
        label_v[zero_index] = 0
        label_v = label_v.T
        label_v = label_v.sum(0)
        Y2 = label_v[unlabeled_index]

        Y1_real = np.array(Y1, copy=True)
        Y1_real[np.where(Y1 == -1)] = 0
        print()
        #Y3 = label_v[labeled_index]
        Y3_real = np.array(Y1, copy=True)
        print("thre2",thre)
        n=n+1
        thre = get_thre(Y3_real, Y4, 2*thre,n)
        #thre = get_thre_by_acc(np.mat(Y1_real), np.mat(Y1_pro))
        #thre=0.5
        print("thre1",thre)
        max_Y2 = np.max(Y2)
        min_Y2 = np.min(Y2)
        Y2 = (Y2 - min_Y2) / (max_Y2 - min_Y2)
        Y2[np.where(Y2 >= thre)] = 1
        Y2[np.where(Y2 < thre)] = -1
        Y4[np.where(Y4 >= thre)] = 1
        Y4[np.where(Y4 < thre)] = -1
        '''elif t_method == 'Y1+w+transmit+highestACC_transmitY2':
            label_c = np.array(label, copy=True)
            label_c[unlabeled_index] = Y2
            label_c[labeled_index] = Y1_pro
            zero_index = np.where(news_user_matrix == 0)
            label_v = (np.multiply(news_user_matrix, Wc).T + label_c).T
            label_v[zero_index] = 0
            label_v = label_v.T
            label_v = label_v.sum(0)
            label_v = 1.0 - np.cos(0.1 * label_v)
            Y2 = label_v[unlabeled_index]
            Y1_pro = label_v[labeled_index]
            Y1_real = np.array(Y1, copy=True)
            Y1_real[np.where(Y1 == -1)] = 0
            thre = get_thre_by_acc(np.mat(Y1_real), np.mat(Y1_pro))
            print()
            print("外面的",thre)
            Y2[np.where(Y2 >= thre)] = 1
            Y2[np.where(Y2 < thre)] = -1'''


        Y2_pre = np.array(Y2, copy=True)
        Y2_pre[np.where(Y2_pre == -1)] = 0
        Y2_real = np.array(X2_label, copy=True)
        Y2_real[np.where(Y2_real == -1)] = 0

        TP = np.dot(Y2_pre, Y2_real)
        FP = np.sum(Y2_pre) - TP
        FN = Y2_real.sum() - TP
        TN = u - TP - FP - FN
        acc_iter_list.append((TP + TN) / u)
        print(TP)
        print(TN)
        print("第",n,"结果",(TP + TN)/u)

        # print(np.sum(Y2_pre))
        # print(np.dot(Y2_pre, Y2_real) / np.sum(Y2_pre))

        label[unlabeled_index] = Y2
        news_user_label_mat = np.multiply(label, news_user_matrix.T).T
        user_news_count = news_user_matrix.sum(0)
        user_truth_news_count = (np.abs(news_user_label_mat).sum(0) + news_user_label_mat.sum(0)) / 2
        Wc = user_truth_news_count / user_news_count
        Wc, P, Q, F, R = kalman(x, Wc, P, Q, F, R)
        #Wc2=alph_filter(Wc2,Wc3)
        # Y3 = np.append(Y1, Y2)
        X_add = np.vstack([X2[po_index_train], X2[ne_index_train]])
        sample_train_po_label = np.ones(sample_train)
        y_add = np.append(sample_train_po_label, -sample_train_po_label)
        X3 = np.vstack([X1, X_add])
        Y3 = np.append(Y1, y_add)
        #if i==20:
           # np.savetxt("y.txt",acc_iter_list,fmt="%f", delimiter=",")
    return acc_iter_list


def self_train_SVM_without_w(X1, Y1, X2, X2_label, C=1, kernel='linear', max_epoch=100):
    l = X1.shape[0]
    u = X2.shape[0]
    N = l + u
    acc_iter_list = []

    clf = svm.SVC(C=10, kernel='rbf', degree=0.1, gamma=10, coef0=0.0,
                  shrinking=True, probability=True, tol=0.001, cache_size=200,
                  class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
                  random_state=None)
    clf.fit(X1, Y1)
    Y2 = clf.predict(X2)

    print()
    print(np.where(X2_label == 1)[0].shape)
    print(np.where(X2_label == -1)[0].shape)
    print(np.where(Y2 == 1)[0].shape)
    print(np.where(Y2 == -1)[0].shape)

    Y2_pre = np.array(Y2, copy=True)
    Y2_pre[np.where(Y2_pre == -1)] = 0
    Y2_real = np.array(X2_label, copy=True)
    Y2_real[np.where(Y2_real == -1)] = 0

    TP = np.dot(Y2_pre, Y2_real)
    FP = np.sum(Y2_pre) - TP
    FN = Y2_real.sum() - TP
    TN = u - TP - FP - FN
    acc_iter_list.append((TP + TN) / u)
    print((TP + TN) / u)
    X3 = np.vstack([X1, X2])
    Y3 = np.append(Y1, Y2)

    for i in range(max_epoch):
        # clf = svm.SVC(C=C, kernel=kernel, probability=True)
        clf.fit(X3, Y3)
        Y2 = clf.predict(X2)
        print()
        print(np.where(Y2 == 1)[0].shape)
        print(np.where(Y2 == -1)[0].shape)

        Y2_pre = np.array(Y2, copy=True)
        Y2_pre[np.where(Y2_pre == -1)] = 0
        Y2_real = np.array(X2_label, copy=True)
        Y2_real[np.where(Y2_real == -1)] = 0

        TP = np.dot(Y2_pre, Y2_real)
        FP = np.sum(Y2_pre) - TP
        FN = Y2_real.sum() - TP
        TN = u - TP - FP - FN
        acc_iter_list.append((TP + TN) / u)
        print((TP + TN) / u)
        Y3 = np.append(Y1, Y2)
    return acc_iter_list


def TSVM_train(X1, Y1, X2, X2_label, Cu=0.001, Cl=1, kernel='linear',):
        # Train TSVM by X1, Y1, X2
        # Parameters
        # ----------
        # X1: Input data with labels
        #         np.array, shape:[n1, m], n1: numbers of samples with labels, m: numbers of features
        # Y1: labels of X1
        #         np.array, shape:[n1, ], n1: numbers of samples with labels
        # X2: Input data without labels
        #         np.array, shape:[n2, m], n2: numbers of samples without labels, m: numbers of features

        l = X1.shape[0]
        u = X2.shape[0]
        N = l + u
        sample_weight = np.ones(N)
        sample_weight[l:] = Cu

        clf = svm.SVC(C=10, kernel='rbf', degree=0.1, gamma=10, coef0=0.0,
                      shrinking=True, probability=True, tol=0.001, cache_size=200,
                      class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
                      random_state=None)
        clf.fit(X1, Y1)
        Y2 = clf.predict(X2)

        Y2_pre = np.array(Y2, copy=True)
        Y2_pre[np.where(Y2_pre == -1)] = 0
        Y2_real = np.array(X2_label, copy=True)
        Y2_real[np.where(Y2_real == -1)] = 0
        print()
        TP = np.dot(Y2_pre, Y2_real)
        FP = np.sum(Y2_pre) - TP
        FN = Y2_real.sum() - TP
        TN = u - TP - FP - FN
        print((TP + TN) / u)

        # Y2 = np.expand_dims(Y2, 1)
        X2_id = np.arange(u)
        X3 = np.vstack([X1, X2])
        Y3 = np.append(Y1, Y2)

        while Cu < Cl:

            clf.fit(X3, Y3, sample_weight=sample_weight)  # overwrite whole model

            Y2 = clf.predict(X2)
            Y2_pre = np.array(Y2, copy=True)
            Y2_pre[np.where(Y2_pre == -1)] = 0
            Y2_real = np.array(X2_label, copy=True)
            Y2_real[np.where(Y2_real == -1)] = 0
            print()
            TP = np.dot(Y2_pre, Y2_real)
            FP = np.sum(Y2_pre) - TP
            FN = Y2_real.sum() - TP
            TN = u - TP - FP - FN
            print((TP + TN) / u)

            while True:
                Y2_d = clf.decision_function(X2)    # linear: w^Tx + b
                Y2 = Y2.reshape(-1)
                epsilon = 1 - Y2 * Y2_d   # calculate function margin
                positive_set, positive_id = epsilon[Y2 > 0], X2_id[Y2 > 0]
                negative_set, negative_id = epsilon[Y2 < 0], X2_id[Y2 < 0]
                positive_max_id = positive_id[np.argmax(positive_set)]
                negative_max_id = negative_id[np.argmax(negative_set)]
                a, b = epsilon[positive_max_id], epsilon[negative_max_id]
                if a > 0 and b > 0 and a + b > 2.0:
                    Y2[positive_max_id] = Y2[positive_max_id] * -1
                    Y2[negative_max_id] = Y2[negative_max_id] * -1
                    # Y2 = np.expand_dims(Y2, 1)
                    Y3 = np.append(Y1, Y2)
                    clf.fit(X3, Y3, sample_weight=sample_weight)

                    Y2 = clf.predict(X2)
                    # Y2_pre = np.array(Y2, copy=True)
                    # Y2_pre[np.where(Y2_pre == -1)] = 0
                    # Y2_real = np.array(X2_label, copy=True)
                    # Y2_real[np.where(Y2_real == -1)] = 0
                    # print()
                    # TP = np.dot(Y2_pre, Y2_real)
                    # FP = np.sum(Y2_pre) - TP
                    # FN = Y2_real.sum() - TP
                    # TN = u - TP - FP - FN
                    # print((TP + TN) / u)

                else:
                    break
            Cu = min(2*Cu, Cl)
            sample_weight[l:] = Cu



def kalman(x,y,P,Q,F,R):

    x=F*x#公式1
    e=math.sqrt(math.pow(P,2)+math.pow(Q,2))#公式2
    K = math.pow(e, 2) / (math.pow(e, 2) + math.pow(R, 2))  # 公式3
    c = y - x  # 残差
    x=x+K*c#公式4
    P = math.sqrt((1 - math.sqrt(K)) * math.pow(e, 2))  # 公式5
    return x,P,Q,F,R

def get_thre_by_acc(real_score, predict_score):
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))

    thresholds = np.mat(sorted_predict_score)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1

    TP = predict_score_matrix * real_score.T
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    # fpr = FP / (FP + TN)
    # tpr = TP / (TP + FN)
    # ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    # ROC_dot_matrix.T[0] = [0, 0]
    # ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    # x_ROC = ROC_dot_matrix[0].T
    # y_ROC = ROC_dot_matrix[1].T
    # auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    # recall_list = tpr
    # precision_list = TP / (TP + FP)
    # PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, precision_list)).tolist())).T
    # PR_dot_matrix.T[0] = [0, 1]
    # PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    # x_PR = PR_dot_matrix[0].T
    # y_PR = PR_dot_matrix[1].T
    # aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    # f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    print("accuracy_list",accuracy_list)
    print(len(accuracy_list))
    # specificity_list = TN / (TN + FP)

    # max_index = np.argmax(f1_score_list)
    max_index = np.argmax(accuracy_list)
    # print("Y1：")
    #
    # print(TP[max_index, 0] + FP[max_index, 0])
    # print(FN[max_index, 0] + TN[max_index, 0])
    # print(accuracy_list[max_index, 0])

    # max_index = np.argmax(precision_list)
    threshold = thresholds[0, max_index]

    # f1_score = f1_score_list[max_index, 0]
    # accuracy = accuracy_list[max_index, 0]
    # specificity = specificity_list[max_index, 0]
    # recall = recall_list[max_index, 0]
    # precision = precision_list[max_index, 0]

    return threshold

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

#l=self_train_SVM(X1, Y1, X2, label, labeled_index, unlabeled_index, news_user_matrix, X2_label, C=1, kernel='linear', max_epoch=100)
l=self_train_SVM(X1, Y1, X2, label, labeled_index, unlabeled_index, news_user_matrix, X2_label, C=1, kernel='linear', max_epoch=100)
a=open("result_with_user_textcnn.txt","w")
for i in range(len(l)):

        s = str(l[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择

        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符

        a.write(s)
a.close()
#np.savetxt('result_with_user_初始阈值为0.9.txt',l,fmt="%f", delimiter=",")
#print(type(l))
print(l)
# self_train_SVM_without_w(X1, Y1, X2, X2_label, C=20, kernel='linear', max_epoch=50)
# np.savetxt('acc_without_w.txt', np.array(acc_without_w))
# acc_w_thre_without_sam = self_train_SVM(X1, Y1, X2, label, labeled_index, unlabeled_index, news_user_matrix, X2_label, C=5, kernel='linear', max_epoch=500, t_method='Y1+w+highestACC_without_normal')
# np.savetxt('acc_w_thre_without_sam_0.05.txt', np.array(acc_w_thre_without_sam))
# news_user_matrix = np.multiply(label, news_user_matrix.T).T
# t_de_method = ['Y1+w+highestACC_without_normal', 'Y1+highestACC_with_max_min_normal', 'Y1+w+transmit+highestACC_transmitY2', 'Y1+highestACC_transmitY2']
# count = 0
# for m in t_de_method:
#     count = count + 1
#     precision_list = self_train_SVM(X1, Y1, X2, label, labeled_index, unlabeled_index, news_user_matrix, X2_label, C=1, kernel='linear', max_epoch=50, t_method=m)
#     file_name = 'precision_list_' + str(count) + '.txt'
#     np.savetxt(file_name, np.array(precision_list))
# a = np.array([2,3,4,5])
# print(np.median(a))
# b = np.array([2,5,4,8])
# c = np.append(a,b)

# np.savetxt('precision_list.txt', np.array(precision_list))






