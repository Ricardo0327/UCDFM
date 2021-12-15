
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop
import keras
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.callbacks import EarlyStopping
#!/usr/bin/python3

#
#  Copyright 2016-2018 Peter de Vocht
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import spacy
import math
import gensim
import numpy as np
from sklearn.decomposition import PCA
from typing import List

# see spacy_sentence2vec.py for an example usage with real language inputs


# an embedding word with associated vector
class Word:
    def __init__(self, text, vector):
        self.text = text
        self.vector = vector

    def __str__(self):
        return self.text + ' : ' + str(self.vector)

    def __repr__(self):
        return self.__str__()


# a sentence, a list of words
class Sentence:
    def __init__(self, word_list):
        self.word_list = word_list

    # return the length of a sentence
    def len(self) -> int:
        return len(self.word_list)

    def __str__(self):
        word_str_list = [word.text for word in self.word_list]
        return ' '.join(word_str_list)

    def __repr__(self):
        return self.__str__()


# todo: get a proper word frequency for a word in a document set
# or perhaps just a typical frequency for a word from Google's n-grams
def get_word_frequency(word_text):
    return 0.0001  # set to a low occurring frequency - probably not unrealistic for most words, improves vector values


# A SIMPLE BUT TOUGH TO BEAT BASELINE FOR SENTENCE EMBEDDINGS
# Sanjeev Arora, Yingyu Liang, Tengyu Ma
# Princeton University
# convert a list of sentence with word2vec items into a set of sentence vectors
def sentence_to_vec(sentence_list: List[Sentence], embedding_size: int, a: float=1e-3):
    sentence_set = []
    for sentence in sentence_list:
        vs = np.zeros(embedding_size)  # add all word2vec values into one vector for the sentence
        sentence_length = sentence.len()
        for word in sentence.word_list:
            a_value = a / (a + get_word_frequency(word.text))  # smooth inverse frequency, SIF
            vs = np.add(        vs, np.multiply(a_value, word.vector))  # vs += sif * word_vector

        vs = np.divide(vs, sentence_length)  # weighted average
        sentence_set.append(vs)  # add to our existing re-calculated set of sentences

    # calculate PCA of this sentence set
    pca = PCA()
    pca.fit(np.array(sentence_set))
    u = pca.components_[0]  # the PCA vector
    u = np.multiply(u, np.transpose(u))  # u x uT

    # pad the vector?  (occurs if we have less sentences than embeddings_size)
    if len(u) < embedding_size:
        for i in range(embedding_size - len(u)):
            u = np.append(u, 0)  # add needed extension for multiplication below

    # resulting sentence vectors, vs = vs -u x uT x vs
    sentence_vecs = []
    for vs in sentence_set:
        sub = np.multiply(u,vs)
        sentence_vecs.append(np.subtract(vs, sub))

    return sentence_vecs
# use the spacy large model's vectors for testing semantic relatedness
# this assumes you've already installed the large model, if not download it and pip install it:
# wget https://github.com/explosion/spacy-models/releases/tag/en_core_web_lg-2.0.0
# pip install en_core_web_lg-2.0.0.tar.gz
# nlp = spacy.load('en_core_web_lg')



# euclidean distance between two vectors
def l2_dist(v1, v2):
    sum = 0.0
    if len(v1) == len(v2):
        for i in range(len(v1)):
            delta = v1[i] - v2[i]
            sum += delta * delta
        return math.sqrt(sum)

def model(train, train_label):
    train = train.reshape(train.shape[0], -1, 1)
    ohe = OneHotEncoder()
    train_label = ohe.fit_transform(np.array(train_label).reshape(-1, 1)).toarray()
    ## 定义LSTM模型
    inputs = Input(name='inputs', shape=[train.shape[1], train.shape[2]])
    ## Embedding(词汇表大小,batch大小,每个新闻的词长)
    layer = LSTM(128)(inputs)
    layer = Dense(256, activation="relu", name="FC1")(layer)
    # layer = Dropout(0.1)(layer)
    layer = Dense(2, activation="softmax", name="FC2")(layer)
    model = Model(inputs=inputs, outputs=layer)
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer=RMSprop(), metrics=["accuracy"])
    model.fit(train, train_label, batch_size=128, epochs=10,
              validation_data=(train, train_label),
              callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20)]
              ## 当val-loss不再提升时停止训练
              )
    return model


if __name__ == '__main__':
    x=input()
    embedding_size = 32   # dimension of word embeddings
    wordVec = gensim.models.KeyedVectors.load_word2vec_format("word2vec.txt")

    sentences = []
    #with open('w2v.txt') as reader:#w2v为英文句子
    for z in x:
        if len(z.strip()) > 0:
            sentences.append(z.strip().split(' '))

    # convert the above sentences to vectors using spacy's large model vectors
    sentence_list = []
    for sentence in sentences:
        word_list = []
        for word in sentence:
            # token = nlp.vocab[word]
            if word in wordVec:  # ignore OOVs
                word_list.append(Word(word, wordVec[word]))
        if len(word_list) > 0:  # did we find any words (not an empty set)
            sentence_list.append(Sentence(word_list))

    # apply single sentence word embedding
    sentence_vector_lookup = dict()
    sentence_vectors = np.array(sentence_to_vec(sentence_list, embedding_size))
    #np.savetxt("1.txt", sentence_vectors)
    # all vectors converted together
    if len(sentence_vectors) == len(sentence_list):
        for i in range(len(sentence_vectors)):
            # map: text of the sentence -> vector
            sentence_vector_lookup[sentence_list[i].__str__()] = sentence_vectors[i]
    y = sentence_vectors
    y=y[0]
    y=np.array(y)
    print(type(y))
    y=y.reshape(1,32)
    np.savetxt("y_vector.txt", y, fmt="%f", delimiter=",")
    print(y.shape)







    # where user_news_matrix_i_j=1 means the user j posts the news i
    label = np.loadtxt('label_1.txt')
    print(np.where(label == 1)[0].shape)
    print(np.where(label == 0))
    print(np.where(label == -1)[0].shape)

    news_num = 8419
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
    
    Y1[np.where(Y1 == -1)] = 0
    X2_label[np.where(X2_label == -1)] = 0
    print(Y1)
    print(X2_label)
    train=X1
    train_label=Y1
    train = train.reshape(train.shape[0], -1, 1)

    ohe = OneHotEncoder()
    train_label = ohe.fit_transform(np.array(train_label).reshape(-1, 1)).toarray()
    ## 定义LSTM模型
    inputs = Input(name='inputs', shape=[train.shape[1], train.shape[2]])
    ## Embedding(词汇表大小,batch大小,每个新闻的词长)
    layer = LSTM(128)(inputs)
    layer = Dense(256, activation="relu", name="FC1")(layer)
    # layer = Dropout(0.1)(layer)
    layer = Dense(2, activation="softmax", name="FC2")(layer)
    model = Model(inputs=inputs, outputs=layer)
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer=RMSprop(), metrics=["accuracy"])
    model.fit(train, train_label, batch_size=128, epochs=10,
              validation_data=(train, train_label),
              callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20)]
              ## 当val-loss不再提升时停止训练
              )
    Y2 = model.predict(y)
    predict = Y2[:, 1]
    # encode_text=encode_text.reshape(-1,32)
    # print(encode_text)
    # predict = np.argmax(encode_text, axis=1)
    # predict= encode_text[:,1]
    # np.savetxt('predict1.txt',predict,fmt="%f", delimiter=",")
    if predict > 0.5:
            label = 1
    if predict < 0.5:
            label = 0


    print("the label of this news is",label)





