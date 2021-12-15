import numpy as np
import sklearn.svm as svm
from sklearn.metrics import accuracy_score
from filterpy.kalman import KalmanFilter
import math
from  sklearn.cluster  import  KMeans
from scipy.stats import pearsonr
def cluster(dataset,k):
    dataSet = np.mat(dataset)
    #dataSet = dataSet.reshape(-1, 1)用于一维向量
    estimator = KMeans(n_clusters=k)  # 构造聚类器
    estimator.fit(dataSet)
    a=np.zeros((8419,20))
    label_pred = estimator.labels_  # 获取聚类标签
    for i in range(k):
      a[label_pred == i,i]=1
    print(a)
    print(a.shape)
    return  a

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
a=cluster(news_vector,20)
print(news_user_matrix.shape)
user_cluster=np.dot(news_user_matrix.T,a)
user_user=np.zeros((1066,1066))
#np.savetxt("user_cluster.txt",user_cluster,fmt="%f", delimiter=",")

print(user_cluster[0])
#pccs = pearsonr(user_cluster[0], user_cluster[1])#结果(a,b)a表示相关系数，b表示相关系数的显著性
#print(pccs)
for i in range(1066):
    for j in range(1066):
        a,b=pearsonr(user_cluster[i], user_cluster[j])
        user_user[i,j]=a
print(user_user.shape)
user_user[np.where(user_user<=0.4)]=0
user_user[np.where(user_user>0.4)]=1
#user_user_relationship=np.zeros(())


news_user_label_mat = np.multiply(label, news_user_matrix.T).T
user_news_count = news_user_matrix.sum(0)
user_truth_news_count = (np.abs(news_user_label_mat).sum(0) + news_user_label_mat.sum(0)) / 2
Wc = user_truth_news_count / user_news_count
n=0
for i in range(1066):
    n+=1
    w=0
    count=0
    for j in range(n,1066,1):
        if user_user[i,j]==1:
            count=count+1
            w=w+user_user[i,j]
            average=w/count
    Wc[np.where(user_user[i])==1] = average
np.savetxt("Wc.txt",Wc,fmt="%f", delimiter=",")
print(Wc)
print(111111)

#print(user.shape)
