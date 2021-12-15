import numpy as np
import sklearn.svm as svm
from sklearn.metrics import accuracy_score


def self_train(X1, Y1, X2, label, labeled_index, unlabeled_index, news_user_matrix, X2_label, C=1, kernel='linear', max_epoch=200):
    l = X1.shape[0]
    u = X2.shape[0]
    N = l + u
    news_user_label_mat = np.multiply(label, news_user_matrix.T).T
    user_news_count = news_user_matrix.sum(0)
    user_truth_news_count = (np.abs(news_user_label_mat).sum(0) + news_user_label_mat.sum(0)) / 2
    unknown_any_label_index = np.where(np.abs(news_user_label_mat).sum(0) == 0)
    Wc = user_truth_news_count / user_news_count
    Wc[unknown_any_label_index] = 0.5

    precision_iter_list = []

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

    TP = np.dot(Y2_pre, Y2_real)
    FP = np.sum(Y2_pre) - TP
    FN = Y2_real.sum() - TP
    TN = u - TP - FP - FN
    print()
    print(TP)
    print(TN)
    print((TP + TN) / u)

    # Y2 = clf.predict_proba(X2)[:, 1]
    # label_c = np.array(label, copy=True)
    # label_c[unlabeled_index] = Y2
    # zero_index = np.where(news_user_matrix == 0)
    # label_v = (np.multiply(news_user_matrix, Wc).T + label_c).T
    # label_v[zero_index] = 0
    # label_v = label_v.T
    # label_v = label_v.sum(0) / 2.0
    # Y2 = label_v[unlabeled_index]
    # Y2[np.where(Y2 >= 0.5)] = 1
    # Y2[np.where(Y2 < 0.5)] = -1
    #
    # Y2_pre = np.array(Y2, copy=True)
    # Y2_pre[np.where(Y2_pre == -1)] = 0
    # Y2_real = np.array(X2_label, copy=True)
    # Y2_real[np.where(Y2_real == -1)] = 0
    #
    # TP = np.dot(Y2_pre, Y2_real)
    # FP = np.sum(Y2_pre) - TP
    # FN = Y2_real.sum() - TP
    # TN = u - TP - FP - FN
    # print()
    # print(TP)
    # print(TN)
    # print((TP + TN) / u)

    label[unlabeled_index] = Y2
    news_user_label_mat = np.multiply(label, news_user_matrix.T).T
    user_truth_news_count = (np.abs(news_user_label_mat).sum(0) + news_user_label_mat.sum(0)) / 2
    Wc = user_truth_news_count / user_news_count
    X3 = np.vstack([X1, X2])
    Y3 = np.append(Y1, Y2)

    for i in range(max_epoch):
        clf.fit(X3, Y3)
        Y2 = clf.predict_proba(X2)[:, 1]

        label_c = np.array(label, copy=True)
        label_c[unlabeled_index] = Y2
        zero_index = np.where(news_user_matrix == 0)
        label_v = (np.multiply(news_user_matrix, Wc).T + label_c).T
        label_v[zero_index] = 0
        label_v = label_v.T
        label_v = label_v.sum(0)

        Y2 = label_v[unlabeled_index]/2
        Y2[np.where(Y2 >= 0.5)] = 1
        Y2[np.where(Y2 < 0.5)] = -1

        Y2_pre = np.array(Y2, copy=True)
        Y2_pre[np.where(Y2_pre == -1)] = 0
        Y2_real = np.array(X2_label, copy=True)
        Y2_real[np.where(Y2_real == -1)] = 0

        TP = np.dot(Y2_pre, Y2_real)
        FP = np.sum(Y2_pre) - TP
        FN = Y2_real.sum() - TP
        TN = u - TP - FP - FN
        print()
        print(TP)
        print(TN)
        print((TP + TN)/u)

        label[unlabeled_index] = Y2
        news_user_label_mat = np.multiply(label, news_user_matrix.T).T
        user_truth_news_count = (np.abs(news_user_label_mat).sum(0) + news_user_label_mat.sum(0)) / 2
        Wc = user_truth_news_count / user_news_count
        Y3 = np.append(Y1, Y2)
    return precision_iter_list


def self_train_SVM_without_w(X1, Y1, X2, X2_label, C=1, kernel='linear', max_epoch=200):
    l = X1.shape[0]
    u = X2.shape[0]
    N = l + u
    precision_iter_list = []
    sample_train = int(0.1 * u)

    clf = svm.SVC(C=10, kernel='rbf',gamma=10, probability=True)
    clf.fit(X1, Y1)
    Y2 = clf.predict_proba(X2)[:, 1]
    sort_index = np.argsort(Y2)
    ne_index_train = sort_index[:sample_train]
    po_index_train = sort_index[-sample_train:]
    Y1_pro = clf.predict_proba(X1)[:, 1]
    Y1_real = np.array(Y1, copy=True)
    Y1_real[np.where(Y1 == -1)] = 0
    thre = get_thre_by_acc(np.mat(Y1_real), np.mat(Y1_pro))
    po_index = np.where(Y2 >= thre)
    ne_index = np.where(Y2 < thre)
    Y2[po_index] = 1
    Y2[ne_index] = -1
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
    print((TP + TN) / u)
    X_add = np.vstack([X2[po_index_train], X2[ne_index_train]])
    sample_train_po_label = np.ones(sample_train)
    y_add = np.append(sample_train_po_label, -sample_train_po_label)
    X3 = np.vstack([X1, X_add])
    Y3 = np.append(Y1, y_add)

    for i in range(max_epoch):
        clf.fit(X3, Y3)
        Y2 = clf.predict_proba(X2)[:, 1]
        sort_index = np.argsort(Y2)
        ne_index_train = sort_index[:sample_train]
        po_index_train = sort_index[-sample_train:]

        Y1_pro = clf.predict_proba(X1)[:, 1]
        Y1_real = np.array(Y1, copy=True)
        Y1_real[np.where(Y1 == -1)] = 0
        thre = get_thre_by_acc(np.mat(Y1_real), np.mat(Y1_pro))
        po_index = np.where(Y2 >= thre)
        ne_index = np.where(Y2 < thre)
        Y2[po_index] = 1
        Y2[ne_index] = -1
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
        print((TP + TN) / u)
        X_add = np.vstack([X2[po_index_train], X2[ne_index_train]])
        sample_train_po_label = np.ones(sample_train)
        y_add = np.append(sample_train_po_label, -sample_train_po_label)
        X3 = np.vstack([X1, X_add])
        Y3 = np.append(Y1, y_add)


    return precision_iter_list


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


def sigmoid(x):
    return 1/(1 + np.exp(-x))


if __name__ == "__main__":
    news_user_matrix = np.loadtxt('news_user_matrix_1.txt')

    # where user_news_matrix_i_j=1 means the user j posts the news i
    label = np.loadtxt('label_1.txt')
    print(np.where(label == 1)[0].shape)
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
    print(np.where(Y1 == 1)[0].shape)
    X2 = news_vector[unlabeled_index]
    self_train_SVM_without_w(X1, Y1, X2, X2_label, C=20, kernel='linear', max_epoch=50)
    # self_train(X1, Y1, X2, label, labeled_index, unlabeled_index, news_user_matrix, X2_label, C=1, kernel='linear',
    #                max_epoch=500)
