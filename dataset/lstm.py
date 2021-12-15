
# coding: utf-8

# In[64]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping




train=open("train.txt",encoding="utf-8").readlines()
train=[i.strip().split(",") for i in train]
train=np.array(train)

test=open("test.txt",encoding="utf-8").readlines()
test=[i.strip().split(",") for i in test]
import numpy as np
test=np.array(test)

train_label=[i.strip() for i in open("train_label.txt",encoding="utf-8").readlines()]
test_label=[i.strip() for i in open("test_label.txt",encoding="utf-8").readlines()]
train=train.reshape(train.shape[0],-1,1)
test=test.reshape(test.shape[0],-1,1)

ohe = OneHotEncoder()
train_label= ohe.fit_transform(np.array(train_label).reshape(-1,1)).toarray()
test_label = ohe.transform(np.array(test_label).reshape(-1,1)).toarray()


# In[71]:


# from sklearn import datasets
# import matplotlib.pyplot as plt
 
# #加载数据集，是一个字典类似Java中的map
# lris_df = datasets.load_iris()
# train=lris_df["data"]
# target=lris_df["target"]




# from sklearn import datasets
# import pandas as pd
# from sklearn.model_selection import train_test_split
# '''载入数据'''
# X,y = datasets.load_iris(return_X_y=True)

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

# tmp=np.where((y_train==0)| (y_train==1))
# X_train=X_train[tmp]
# y_train=y_train[tmp]
# tmp=np.where((y_test==0)| (y_test==1))
# X_test=X_test[tmp]
# y_test=y_test[tmp]

# train=X_train.reshape(X_train.shape[0],-1,1)
# test=X_test.reshape(X_test.shape[0],-1,1)

# ohe = OneHotEncoder()
# train_label= ohe.fit_transform(np.array(y_train).reshape(-1,1)).toarray()
# test_label = ohe.transform(np.array(y_test).reshape(-1,1)).toarray()


# In[80]:
Y2_real=np.loadtxt("test_label.txt")

## 定义LSTM模型
inputs = Input(name='inputs',shape=[train.shape[1],train.shape[2]])
## Embedding(词汇表大小,batch大小,每个新闻的词长)
layer = LSTM(128)(inputs)
layer = Dense(256,activation="relu",name="FC1")(layer)
# layer = Dropout(0.1)(layer)
layer = Dense(2,activation="softmax",name="FC2")(layer)
model = Model(inputs=inputs,outputs=layer)
model.summary()
model.compile(loss="categorical_crossentropy",optimizer=RMSprop(),metrics=["accuracy"])


# In[81]:


model_fit = model.fit(train,train_label,batch_size=128,epochs=1000,
                      validation_data=(test,test_label),
                      callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001, patience=20)] ## 当val-loss不再提升时停止训练
                      )


print ("准确率:",np.mean(np.argmax(model.predict(test),axis=-1)==np.argmax(test_label,axis=-1)))

Y3 = np.argmax(model.predict(test),axis=-1)
print(Y3)
Y2_pre = model.predict(test)
Y2_pre = Y2_pre[:,1]
print(Y2_pre)
np.savetxt("accuracy.txt",Y2_pre,fmt="%f", delimiter=",")

accuracy = metrics.accuracy_score( Y2_real,Y3)
f1 = metrics.f1_score( Y2_real, Y3)
precision = metrics.precision_score( Y2_real, Y3)
recall = metrics.recall_score( Y2_real, Y3)
aucroc = metrics.roc_auc_score( Y2_real, Y3)
print("accuracy",accuracy)
print("f1",f1)
print("precision",precision)
print("recall",recall)
print("aucroc",aucroc)
'''np.savetxt("accuracy.txt",test_accuracy,fmt="%f", delimiter=",")
np.savetxt("test_f1.txt", test_f1, fmt="%f", delimiter=",")
np.savetxt(" test_precision.txt",  test_precision, fmt="%f", delimiter=",")
np.savetxt("test_recall.txt",test_recall, fmt="%f", delimiter=",")
np.savetxt("test_aucroc.txt", test_aucroc, fmt="%f", delimiter=",")'''