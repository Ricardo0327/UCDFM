import numpy as np
import sklearn.svm as svm
from sklearn.metrics import accuracy_score
from filterpy.kalman import KalmanFilter
import math
from  sklearn.cluster  import  KMeans

def cluster(W,k):
    dataSet = np.array(W)
    #dataSet = dataSet.reshape(-1, 1)
    estimator = KMeans(n_clusters=k)  # 构造聚类器
    estimator.fit(dataSet)
    label_pred = estimator.labels_  # 获取聚类标签
    print(label_pred == 0)
    print(label_pred == 1)
    print(label_pred == 2)
    print(label_pred == 3)
    print(label_pred == 4)
    x = 0
    '''for data in dataSet[label_pred == 0]:
        x = x + data
    print(x)
    y = x / (len(dataSet[label_pred == 0]))
    x=0
    for data in dataSet[label_pred == 1]:
        x =  x+ data
    print(x)
    p=  x/ (len(dataSet[label_pred == 1]))
    x=0
    for data in dataSet[label_pred == 2]:
        x = x + data
    print(x)
    q = x / (len(dataSet[label_pred == 2]))
    x=0
    for data in dataSet[label_pred == 3]:
        x = x + data
    print(x)
    f = x / (len(dataSet[label_pred == 3]))
    x=0
    for data in dataSet[label_pred == 4]:
        x = x + data
    print(x)
    m = x / (len(dataSet[label_pred == 4]))
    x=0
    print(p)
    dataSet[label_pred == 0] = y
    dataSet[label_pred == 1] = p
    dataSet[label_pred == 2] = q
    dataSet[label_pred == 3] = f
    dataSet[label_pred == 4] = m
    print('k均值聚类')
    #print(dataSet[label_pred == 0])
    print(np.shape(dataSet))
    dataSet=dataSet.reshape(-1)
    print(np.shape(dataSet))
    return dataSet'''

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
user_news_num=np.loadtxt('user_list_1.txt', delimiter = "," ,usecols=(1) , dtype=float)
# pre-trained news embedding vectors
X1 = news_vector[labeled_index]
print(X1.shape)
Y1 = label[labeled_index]
print(Y1.shape)
print(Y1.sum())
X2 = news_vector[unlabeled_index]
print(np.where(X2_label == 1)[0].shape)
print(np.where(X2_label == -1)[0].shape)
news=np.mat(news_vector)
news_user= np.dot( news_user_matrix.T,news)
#print(news_user_label_mat.shape)
#np.savetxt('用户新闻总和矩阵',news_user_label_mat,fmt="%f", delimiter=",")#
user=1/user_news_num
print(111111)
print(user.shape)
user=(np.mat(user)).T
news_user_average=np.multiply(user,news_user)
np.savetxt('用户新闻平均值矩阵',news_user_average,fmt="%f", delimiter=",")
print(news_user_average.shape)
#print(user.shape)
print(cluster(news_user_average,5))