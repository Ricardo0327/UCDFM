from sklearn.linear_model import LogisticRegression
import numpy as np
import sklearn.svm as svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.svm as svm
train=np.loadtxt('trainVector.txt')
train_label=np.loadtxt('trainLabel.txt')
test=np.loadtxt('testVector.txt')
test_label=np.loadtxt('testLabel.txt')
print(type(train))
print(type(train_label))
#label[np.where(label==-1)]=0
#h=h.reshape(1,len(h)*np.prod(h.shape[1:]))
#index = np.arange(len(h))
#np.random.seed(0)
#np.random.shuffle(index)
#unlabeled_index = index[: int(len(h) * 0.8)]
#abeled_index = index[int(len(h) * 0.8):]
#print(len(labeled_index))
#print(len(unlabeled_index))
#train=h[labeled_index]
#test=h[unlabeled_index]
#train_label=label[labeled_index]
#test_label=label[unlabeled_index]
#test_label1=label[unlabeled_index]
#clf = RandomForestClassifier()
#clf= LogisticRegression(solver='lbfgs')
clf = svm.SVC(C=10, kernel='rbf', degree=0.1, gamma=10, coef0=0.0,
                  shrinking=True, probability=True, tol=0.001, cache_size=200,
                  class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
                  random_state=None)
clf.fit(train,train_label)
Y2 = clf.predict(test)
#print(Y2)
accuracy = metrics.accuracy_score( test_label, Y2)
f1 = metrics.f1_score( test_label, Y2)
precision = metrics.precision_score( test_label, Y2)
recall = metrics.recall_score( test_label, Y2)
aucroc = metrics.roc_auc_score( test_label, Y2)
print(accuracy)
print(f1)
print(precision)
print(recall)
print(aucroc)