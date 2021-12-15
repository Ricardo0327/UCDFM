import numpy as np
import sklearn.svm as svm
from sklearn.metrics import accuracy_score
from filterpy.kalman import KalmanFilter
import math
from  sklearn.cluster  import  KMeans
from scipy.stats import pearsonr,spearmanr
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras import optimizers
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from sklearn import metrics
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping


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

def cluster(dataset,k):
    dataSet = np.mat(dataset)
    #dataSet = dataSet.reshape(-1, 1)用于一维向量
    estimator = KMeans(n_clusters=k)  # 构造聚类器
    estimator.fit(dataSet)
    a=np.zeros((5498,20))
    label_pred = estimator.labels_  # 获取聚类标签
    for i in range(k):
      a[label_pred == i,i]=1
    print(a)
    print(a.shape)
    return  a
def cluster1(dataset,W,k):
    dataSet = np.mat(dataset)
    #dataSet = dataSet.reshape(-1, 1)用于一维向量
    estimator = KMeans(n_clusters=k)  # 构造聚类器
    estimator.fit(dataSet)
    label_pred = estimator.labels_  # 获取聚类标签
    for i in range(k):
        x=0
        for data in W[label_pred==i]:
            x=x+data
        y=x/(len(dataSet[label_pred == i]))
        W[label_pred==i]=y
    return W
def model1(train,test,train_label,test_label):
    #train = train.readlines()
    #train = [i.strip().split(",") for i in train]
    train = np.array(train)
    #test = test.readlines()
    #test = [i.strip().split(",") for i in test]
    test = np.array(test)

    #train_label =train_label .readlines()
    #test_label = test_label.readlines()
    train = train.reshape(train.shape[0], -1, 1)
    test = test.reshape(test.shape[0], -1, 1)

    ohe = OneHotEncoder()
    train_label = ohe.fit_transform(np.array(train_label).reshape(-1, 1)).toarray()
    test_label = ohe.transform(np.array(test_label).reshape(-1, 1)).toarray()
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
    model_fit = model.fit(train, train_label, batch_size=128, epochs=1000,
                          validation_data=(test, test_label),
                          callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20)]
                          ## 当val-loss不再提升时停止训练
                          )
    return  model_fit

def model11(x_train, y_train):
    model = Sequential()
    model.add(Dense(input_dim=10, units=50, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dense(units=25, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dense(units=10, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dense(units=2, activation='softmax', kernel_initializer='random_uniform'))
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy'])
    model.fit(x_train, y_train, batch_size=50, epochs=20, verbose=0)
    return model
def build_encoder(X_train,X_test,train_label,test_label):
    #input=input.reshape(len(input))
    input_size = 32
    hidden_size = 16
    output_size = 32
    x = Input(shape=(input_size,))
    #x = Input(name = 'inputs', shape = [train.shape[1], train.shape[2]])
    #encoder1 = LSTM(32, return_sequences=True) (x)
    # encoder2 = LSTM(32, return_sequences=True)(encoder1)
    #encoder1 = Bidirectional(LSTM(32,return_sequences=True), merge_mode='concat')(x)
    #encoder2 = Bidirectional(LSTM(32, return_sequences=True), merge_mode='concat')(encoder1)
    fc_txt = Dense(32, activation='tanh', name='dense_txt', )(x)
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
    r = Dense(32, activation='tanh')(h)
    h = Dense(64, activation='tanh', )(r)
    h = Dense(32, activation='tanh', )(h)
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
    autoencoder.fit(X_train, train_label, batch_size=1, epochs=10,
              validation_data=(X_test, test_label),
              callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20)])
    conv_encoder = Model(x, h)  # 只取编码器做模型
    encoded_text = conv_encoder.predict(X_test)
    decoded_text = autoencoder.predict(X_test)#编码解码后的
    return decoded_text
def self_train_SVM(X1, Y1, X2, label, labeled_index, unlabeled_index, news_user_matrix, X2_label, C=1, kernel='linear', max_epoch=30):
    l = X1.shape[0]
    u = X2.shape[0]
    N = l + u
    sample_train = int(0.05 * u)
    #X3
    news_user_label_mat = np.multiply(label, news_user_matrix.T).T
    user_news_count = news_user_matrix.sum(0)
    user_truth_news_count = (np.abs(news_user_label_mat).sum(0) + news_user_label_mat.sum(0)) / 2
    unknown_any_label_index = np.where(np.abs(news_user_label_mat).sum(0) == 0)
    print("user_truth_news_count",user_truth_news_count)
    Wc = user_truth_news_count / user_news_count
    Wc[unknown_any_label_index] = 0.5
    print('++++++++++++++++++++++++++++++++++++++++')
    unknown_any_label_index=np.mat(unknown_any_label_index)
    print(unknown_any_label_index.shape)
    x = Wc
    print("Wc", Wc)
    acc_iter_list = []
    print("--------------------------------------------------------------------------")
    '''clf = svm.SVC(C=10, kernel='rbf', degree=0.1, gamma=10, coef0=0.0,
                  shrinking=True, probability=True, tol=0.001, cache_size=200,
                  class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
                  random_state=None)
    clf = svm.SVC(C=10, kernel='rbf', degree=0.1, gamma=10, coef0=0.0,
                  shrinking=True, probability=True, tol=0.001, cache_size=200,
                  class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
                  random_state=None)'''
    #clf = LogisticRegression(solver='lbfgs')
    #clf = RandomForestClassifier()
    #clf.fit(X1, Y1)
    #train = np.array(X1)
    # test = test.readlines()
    # test = [i.strip().split(",") for i in test]
    print(X1.shape)
    print(X2.shape)
    print(Y1.shape)
    print(X2_label.shape)
    test = np.array(X2)
    train = np.array(X1)

    train = train.reshape(train.shape[0], -1, 1)
    test = test.reshape(test.shape[0], -1, 1)

    ohe = OneHotEncoder()
    train_label = ohe.fit_transform(np.array(Y1).reshape(-1, 1)).toarray()
    test_label = ohe.transform(np.array(X2_label).reshape(-1, 1)).toarray()

    print(test.shape)
    print(train.shape)
    print(train_label.shape)
    print(test_label.shape)

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
    clf = model.fit(train, train_label, batch_size=128, epochs=1000,
                          validation_data=(test, test_label),
                          callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20)])
    Y2 = model.predict(test)

    #Y2 = build_encoder(train,test,train_label,test_label)
    Y2 = Y2[:,1]
    #Y2 = clf.predict_proba(X2)[:, 1]
    sort_index = np.argsort(Y2)
    ne_index_train = sort_index[:sample_train]
    po_index_train = sort_index[-sample_train:]
    Y1_pro = model.predict(train)[:, 1]
    # Y1_pre = clf.predict(X1)
    thre = 0
    # if t_method == 'Y1+w+highestACC_without_normal':
    label_c = np.array(label, copy=True)
    label_c[unlabeled_index] = Y2
    label_c[labeled_index] = Y1_pro
    zero_index = np.where(news_user_matrix == 0)
    label_v = (0.7*(np.multiply(news_user_matrix, Wc).T) +0.3*(label_c)).T
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
    #label[unlabeled_index] = Y2
    #label1=label[np.where(label==-1)]=0
    #news_user_label_mat1 = np.multiply(label, news_user_matrix.T).T#不考虑假新闻的矩阵
    news_user_label_mat = np.multiply(label, news_user_matrix.T).T
    user_news_count = news_user_matrix.sum(0)
    user_truth_news_count = (np.abs(news_user_label_mat).sum(0) + news_user_label_mat.sum(0)) / 2
    print("user_truth_news_count",user_truth_news_count)
    Wc = user_truth_news_count / user_news_count

    '''a = cluster(news_vector, 20)
    #a[np.where(a > 2)] = 1
    #print(a)
    #np.savetxt("user_buzzFeed.txt", a, fmt="%d", delimiter=" ")'''
    a=np.loadtxt("user_buzzfeed_matrix.txt")
    print(news_user_matrix.shape)
    user_cluster = np.dot(news_user_matrix.T, a)
    user_user = np.zeros((111, 111))
    for i in range(111):
        for j in range(111):
            a, b = pearsonr(user_cluster[i], user_cluster[j])
            user_user[i, j] = a
    user_user=np.mat(user_user)
    user_user[np.where(user_user <=0.4)] = 0
    user_user[np.where(user_user > 0.4)] = 1
    o = 0
    for i in range(111):
        o += 1
        w = 0
        count = 0
        for j in range(o, 111, 1):
            if user_user[i, j] == 1:
                count = count + 1
                w = w + user_user[i, j]
                average = w / count
        Wc[np.where(user_user[i]) == 1] = average
    #news = np.mat(news_vector)
    #news_user = np.dot(news_user_label_mat.T, news)#用户新闻总和矩阵
    #user = 1 / user_news_num
    #user = (np.mat(user)).T
    #news_user_average = np.multiply(user, news_user)#用户新闻平均值矩阵
    #Wc=cluster(news_user_average,Wc,500)
    '''y = Wc
    P = 0.02
    Q = 0.008
    F = 1.0
    R = 0.01
    Wc, P, Q, F, R = kalman(x,y , P, Q, F, R)'''
    #rint("Wc1",Wc1)
    #print("Wc",Wc.shape)
    #print("Wc1", Wc1.shape)
    #Wc2 = alph_filter(Wc1, Wc)
    #print("Wc2",Wc2)c
    X_add = np.vstack([X2[po_index_train], X2[ne_index_train]])
    sample_train_po_label = np.ones(sample_train)
    y_add = np.append(sample_train_po_label, -sample_train_po_label)
    X1 = np.vstack([X1, X_add])
    Y1 = np.append(Y1, y_add)
    # X3 = np.vstack([X1, X2])
    # Y3 = np.append(Y1, Y2)
    n=0
    test_accuracy = []
    test_f1 = []
    test_precision = []
    test_recall = []
    test_aucroc = []

    for i in range(max_epoch):
        train = np.array(X1)
        # test = test.readlines()
        # test = [i.strip().split(",") for i in test]
        test = np.array(X2)

        # train_label =train_label .readlines()
        # test_label = test_label.readlines()
        train = train.reshape(train.shape[0], -1, 1)
        test = test.reshape(test.shape[0], -1, 1)
        ohe = OneHotEncoder()
        train_label = ohe.fit_transform(np.array(Y1 ).reshape(-1, 1)).toarray()
        test_label = ohe.transform(np.array(X2_label).reshape(-1, 1)).toarray()
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
        clf = model.fit(train, train_label, batch_size=128, epochs=1000,
                        validation_data=(test, test_label),
                        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20)])
        Y2 = model.predict(test)
        Y2 = Y2[:, 1]

        # Y2_pre = clf.predict(X2)
        # Y2_pre[np.where(Y2_pre == -1)] = 0
        # Y2_real = np.array(X2_label, copy=True)
        # Y2_real[np.where(Y2_real == -1)] = 0
        # prec = precision_score(Y2_real, Y2_pre)
        # print(prec)
        # precision_iter_list.append(prec)

        #Y2 = clf.predict_proba(X2)[:, 1]
        sort_index = np.argsort(Y2)
        ne_index_train = sort_index[:sample_train]
        po_index_train = sort_index[-sample_train:]
        Y1_pro = model.predict(train)[:, 1]
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
        label_v = (0.7*(np.multiply(news_user_matrix, Wc).T) + 0.3*(label_c)).T
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
        Y11 = np.argmax(model.predict(train), axis=-1)
        #thre = get_thre(Y3_real, Y4, 2*thre,n)
        thre =metrics.accuracy_score( Y1,Y11)
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
        #news = np.mat(news_vector)#新闻
        #news_user = np.dot(news_user_matrix.T, news)#每个用户新闻向量的综合
        user_news_count = news_user_matrix.sum(0)
        user_truth_news_count = (np.abs(news_user_label_mat).sum(0) + news_user_label_mat.sum(0)) / 2
        Wc = user_truth_news_count / user_news_count
        o=0
        for i in range(111):
            o += 1
            w = 0
            count = 0
            for j in range(o, 111, 1):
                if user_user[i, j] == 1:
                    count = count + 1
                    w = w + user_user[i, j]
                    average = w / count
            Wc[np.where(user_user[i]) == 1] = average
        #Wc, P, Q, F, R = kalman(x, Wc, P, Q, F, R)
        #Wc2=alph_filter(Wc2,Wc3)
        # Y3 = np.append(Y1, Y2)
        X_add = np.vstack([X2[po_index_train], X2[ne_index_train]])
        sample_train_po_label = np.ones(sample_train)
        y_add = np.append(sample_train_po_label, -sample_train_po_label)
        X1 = np.vstack([X1, X_add])
        Y1 = np.append(Y1, y_add)
        accuracy = metrics.accuracy_score(Y2_real, Y2_pre)
        f1 = metrics.f1_score(Y2_real, Y2_pre)
        precision = metrics.precision_score(Y2_real, Y2_pre)
        recall = metrics.recall_score(Y2_real, Y2_pre)
        aucroc = metrics.roc_auc_score(Y2_real, Y2_pre)
        #T=Y2_pre[np.where(Y2_pre==1)].shape
        #F = Y2_pre[np.where(Y2_pre == 0)].shape
        #print("F__________________________________________________",F)
        #print("T_________________________________________________", T)
        #T=list(T)
        #F = list(F)
        #T.append(T)
        #F.append(F)
        test_accuracy.append(accuracy)
        test_f1.append(f1)
        test_precision.append(precision)
        test_recall.append(recall)
        test_aucroc.append(aucroc)
    #F=np.mat(F)
    #T = np.mat(T)
    test_accuracy = np.mat(test_accuracy)
    test_f1 = np.mat(test_f1)
    test_precision = np.mat(test_precision)
    test_recall = np.mat(test_recall)
    test_aucroc = np.mat(test_aucroc)
    #np.savetxt("model_new_T_0.7.txt", T, fmt="%f", delimiter=",")
    #np.savetxt("model_new_F_0.7.txt", F, fmt="%f", delimiter=",")
    np.savetxt("model_buzzfeed_boltzmann_without_clust.txt",test_accuracy,fmt="%f", delimiter=",")
    np.savetxt("model_buzzfeed_f1_boltzmann_without_clust.txt", test_f1, fmt="%f", delimiter=",")
    np.savetxt("model_buzzfeed_precision_boltzmann_without_clust.txt" , test_precision, fmt="%f", delimiter=",")
    np.savetxt("model_buzzfeed_recall_boltzmann_without_clust.txt",test_recall, fmt="%f", delimiter=",")
    np.savetxt("model_buzzfeed_aucroc_boltzmann_without_clust.txt", test_aucroc, fmt="%f", delimiter=",")
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
    X1 = np.vstack([X1, X2])
    Y1 = np.append(Y1, Y2)

    for i in range(max_epoch):
        # clf = svm.SVC(C=C, kernel=kernel, probability=True)
        clf.fit(X1, Y1)
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
news_user_matrix = np.loadtxt('news_user_matrix_4.txt')

# where user_news_matrix_i_j=1 means the user j posts the news i
label = np.loadtxt('label_4.txt')
print(len(label))
print(np.where(label == 1)[0].shape)
print(np.where(label == 0))
print(np.where(label == -1)[0].shape)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
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
news_vector = np.loadtxt('sentence2Vec_4.txt')
print(news_vector.shape)
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
# pre-trained news embedding vectors
X1 = news_vector[labeled_index]
print(X1.shape)
Y1 = label[labeled_index]
print(np.where(Y1 == 1)[0].shape)
print(np.where(Y1 == -1)[0].shape)
X2 = news_vector[unlabeled_index]
print(np.where(X2_label == 1)[0].shape)
print(np.where(X2_label == -1)[0].shape)
n=0.3
print(X1.shape)
print(Y1.shape)
print(X2.shape)
print(label.shape)
print("------------------------------------")
#l=self_train_SVM(X1, Y1, X2, label, labeled_index, unlabeled_index, news_user_matrix, X2_label, C=1, kernel='linear', max_epoch=30)
l=self_train_SVM(X1, Y1, X2, label, labeled_index, unlabeled_index, news_user_matrix, X2_label, C=1, kernel='linear', max_epoch=20)
'''for i in range(7):
    n=n+0.1
    l=self_train_SVM(X1, Y1, X2, label, labeled_index, unlabeled_index, news_user_matrix, X2_label,n, C=1, kernel='linear', max_epoch=30)
a=open("result_with_user_rf用户内容聚类20.txt","w")
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

# np.savetxt('precision_list.txt', np.array(precision_list))'''






