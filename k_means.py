import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.cluster  import  KMeans

dataSet=[1,2,3,11,12,13]
dataSet=np.array(dataSet)
dataSet=dataSet.reshape(-1,1)
estimator = KMeans(n_clusters=2)#构造聚类器
estimator.fit(dataSet)
label_pred = estimator.labels_ #获取聚类标签
print(dataSet[label_pred==0])
print(dataSet[label_pred==1])
x=0
for data in dataSet[label_pred==0]:
    x=x+data
print(x)
y=x/(len(dataSet[label_pred==0]))
print(y)
dataSet[label_pred==0]=y
print('1111')
print(dataSet[label_pred==0])
print(dataSet)