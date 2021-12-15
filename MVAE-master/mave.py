import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, add
from keras.layers.core import  Dense, Dropout, Activation, Flatten, Reshape
from keras import regularizers
from keras.regularizers import l2
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.utils import np_utils
import pdb
from keras import regularizers
from keras import objectives, backend as K
from keras.layers import Dropout, Reshape, Concatenate, Flatten, Bidirectional, Dense, Embedding, Input, Lambda, LSTM, RepeatVector, TimeDistributed
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam, RMSprop
import keras
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.callbacks import EarlyStopping

def build_encoder(X_train,X_test,train_label,test_label):
    #input=input.reshape(len(input))
    input_size = 60
    hidden_size = 32
    output_size = 32
    x = Input(shape=(input_size,))
    #x = Input(name = 'inputs', shape = [train.shape[1], train.shape[2]])
    #encoder1 = LSTM(32, return_sequences=True) (x)
    # encoder2 = LSTM(32, return_sequences=True)(encoder1)
    #encoder1 = Bidirectional(LSTM(32,return_sequences=True), merge_mode='concat')(x)
    #encoder2 = Bidirectional(LSTM(32, return_sequences=True), merge_mode='concat')(encoder1)
    fc_txt = Dense(64, activation='tanh', name='dense_txt', )(x)
    #layer = Bidirectional(LSTM(32, return_sequences=False,),merge_mode='concat')(layer)
    #layer = LSTM(32,return_sequences=False, name='lstm_txt_2', activation='tanh',
                                    #kernel_regularizer=regularizers.l2(0.05))(layer)
    #fc_txt = Dense(32, return_sequences=False, activation="relu", name="lstm_text_2")(layer)
    '''lstm_txt_1 = Bidirectional(LSTM(32, return_sequences=True, name='lstm_txt_1', activation='tanh',
                                    kernel_regularizer=regularizers.l2(0.05)), merge_mode='concat')(x)
    lstm_txt_2 = Bidirectional(LSTM(32, return_sequences=False, name='lstm_txt_2', activation='tanh',
                                    kernel_regularizer=regularizers.l2(0.05)), merge_mode='concat')(lstm_txt_1)
    fc_txt = Dense(32, activation='tanh', name='dense_txt', kernel_regularizer=regularizers.l2(0.05))(lstm_txt_2)'''

    #h = Dense(64, name='shared', activation='tanh', kernel_regularizer=regularizers.l2(0.05))(fc_txt)
    h = Dense(hidden_size, activation='relu')(fc_txt)
    #decoder = Bidirectional(LSTM(32), merge_mode='concat')(h)
    #decoder1 = LSTM(32, return_sequences=False)(h)
    #decoder2 = LSTM(32, return_sequences=True)(decoder1)
    #decoded_txt = TimeDistributed(Dense(32, activation='softmax'), name='decoded_txt')(decoder2)
    r = Dense(64, activation='tanh')(h)
    h = Dense(128, activation='tanh', )(r)
    h = Dense(64, activation='tanh', )(h)
    r=Dense(2, activation='sigmoid', name='fnd_output')(h)
    '''dec_fc_txt = Dense(32, name='dec_fc_txt', activation='tanh', kernel_regularizer=regularizers.l2(0.05))(h)
    repeated_context = RepeatVector(32)(dec_fc_txt)
    dec_lstm_txt_1 = LSTM(32, return_sequences=True, activation='tanh', name='dec_lstm_txt_1',
                          kernel_regularizer=regularizers.l2(0.05))(repeated_context)
    dec_lstm_txt_2 = LSTM(32, return_sequences=True, activation='tanh', name='dec_lstm_txt_2',
                          kernel_regularizer=regularizers.l2(0.05))(dec_lstm_txt_1)
    r = TimeDistributed(Dense(32, activation='softmax'), name='decoded_txt')(dec_lstm_txt_2)'''
    autoencoder = Model(inputs=x, outputs=r)
    autoencoder.compile(optimizer='adam', loss='mse', metrics=["accuracy"])
    epochs = 1
    batch_size = 2
    '''autoencoder.fit(X_train, X_train,
                              batch_size=batch_size,
                              epochs=epochs, verbose=1,
                              validation_data=(X_test, X_test)
                              )'''
    autoencoder.fit(X_train, train_label, batch_size=1, epochs=20,
              validation_data=(X_test, test_label),
              callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20)])
    conv_encoder = Model(x, h)  # 只取编码器做模型
    encoded_text = conv_encoder.predict(X_test)
    decoded_text = autoencoder.predict(X_test)#编码解码后的
    return decoded_text
def build_fnd( encoded):
    h = Dense(64, activation='tanh', kernel_regularizer=regularizers.l2(0.05))(encoded)
    h = Dense(32, activation='tanh', kernel_regularizer=regularizers.l2(0.05))(h)
    return Dense(1, activation='sigmoid', name='fnd_output')(h)




h=np.loadtxt('1.txt')
label=np.loadtxt('1_label.txt')

label[np.where(label==-1)]=0
#h=h.reshape(1,len(h)*np.prod(h.shape[1:]))
index = np.arange(len(h))
np.random.seed(0)
np.random.shuffle(index)
unlabeled_index = index[: int(len(h) * 0.8)]
labeled_index = index[int(len(h) * 0.8):]
print(len(labeled_index))
print(len(unlabeled_index))
train=h[labeled_index]
test=h[unlabeled_index]
train_label=label[labeled_index]
test_label=label[unlabeled_index]
test_label1=label[unlabeled_index]
#train = train.reshape(train.shape[0], -1, 1)
#test = test.reshape(test.shape[0], -1, 1)
#ohe = OneHotEncoder()
#train_label = ohe.fit_transform(np.array(train_label).reshape(-1, 1)).toarray()
#test_label = ohe.transform(np.array(test_label).reshape(-1, 1)).toarray()
print(train.shape)
print(test.shape)
print(train_label.shape)
print(test_label.shape)
np.savetxt('train1.txt',train,fmt="%f", delimiter=",")
np.savetxt('test1.txt',test,fmt="%f", delimiter=",")
np.savetxt('train_label1.txt',train_label,fmt="%d", delimiter=",")
np.savetxt('test_label1.txt',test_label,fmt="%d", delimiter=",")
print('________________________________________________________________________________________')
#print(build_encoder(train,test))
encode_text=build_encoder(train,test,train_label,test_label)
#encode_text=encode_text.reshape(-1,32)
print(encode_text)
predict = np.argmax(encode_text, axis=1)
#predict= encode_text[:,1]
np.savetxt('predict1.txt',predict,fmt="%f", delimiter=",")
for i in range(len(predict)):
    if predict[i]>0.5:
        predict[i]=1
    if predict[i] < 0.5:
        predict[i] = 0
print(predict)
np.savetxt('predict.txt',predict,fmt="%d", delimiter=",")
np.savetxt('test_label.txt',test_label1,fmt="%d", delimiter=",")
print(predict.shape)
print(test_label1.shape)

#np.savetxt('encode_text.txt',encode_text[0],fmt="%f", delimiter=",")
#predict=build_fnd(encode_text)
#print(predict)
#print(predict.shape)
accuracy = metrics.accuracy_score(test_label1, predict)
f1 = metrics.f1_score(test_label1,  predict)
precision = metrics.precision_score(test_label1, predict)
recall = metrics.recall_score(test_label1, predict)
aucroc = metrics.roc_auc_score(test_label1, predict)
print(accuracy)
print(f1)
print(precision)
print(recall)
print(aucroc)