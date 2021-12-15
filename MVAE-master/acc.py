from sklearn import metrics


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