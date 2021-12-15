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
import os


def build_encoder(X_train,X_test):
    #input=input.reshape(len(input))
    input_size = 32
    hidden_size = 16
    output_size = 32
    x = Input(shape=(input_size,))
    lstm_txt_1 = Bidirectional(LSTM(32, return_sequences=True, name='lstm_txt_1', activation='tanh',
                                    kernel_regularizer=regularizers.l2(0.05)), merge_mode='concat')(x)
    lstm_txt_2 = Bidirectional(LSTM(32, return_sequences=False, name='lstm_txt_2', activation='tanh',
                                    kernel_regularizer=regularizers.l2(0.05)), merge_mode='concat')(lstm_txt_1)
    fc_txt = Dense(32, activation='tanh', name='dense_txt', kernel_regularizer=regularizers.l2(0.05))(
        lstm_txt_2)



    h = Dense(hidden_size, activation='relu')(fc_txt)
    r = Dense(output_size, activation='sigmoid')(h)

    autoencoder = Model(inputs=x, outputs=r)
    autoencoder.compile(optimizer='adam', loss='mse')
    epochs = 1000
    batch_size = 128
    autoencoder.fit(X_train, X_train,
                              batch_size=batch_size,
                              epochs=epochs, verbose=1,
                              validation_data=(X_test, X_test)
                              )
    conv_encoder = Model(x, h)  # 只取编码器做模型
    encoded_text = conv_encoder.predict(X_test)
    decoded_text = autoencoder.predict(X_test)#编码解码后的
    return decoded_text
h=np.loadtxt('sentence2Vec_1.txt')
#h=h.reshape(1,len(h)*np.prod(h.shape[1:]))
h_size=32
index = np.arange(len(h))
np.random.seed(0)
np.random.shuffle(index)
unlabeled_index = index[: int(len(h) * 0.8)]
labeled_index = index[int(len(h) * 0.8):]
print(len(labeled_index))
print(len(unlabeled_index))
print(h_size)
train=h[labeled_index]
test=h[unlabeled_index]
print(train.shape)
print(test.shape)
print(build_encoder(train,test))