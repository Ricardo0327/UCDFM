import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 102)
#acc_list_1 = np.loadtxt('result_with_user_初始阈值为0.74加卡尔曼滤波改动.txt')
acc_list_1 = np.loadtxt('result_with_user_tsvm.txt')
acc_list_2 = np.loadtxt('result_with_user_tsvm用户内容聚类5.txt')
#acc_list_3 = np.loadtxt('result_with_user_rf.txt')
#acc_list_4 = np.loadtxt('result_with_user_lr.txt')
#acc_list_5 = np.loadtxt('result_with_user_0.5.txt')
#acc_list_6 = np.loadtxt('result_with_user_初始阈值为0.5.txt')

plt.plot(x, acc_list_1, color='blue',)
plt.plot(x, acc_list_2, color='r')
#plt.plot(x, acc_list_3, color='g',)
#plt.plot(x, acc_list_4, color='black',)
#plt.plot(x, acc_list_5, color='yellow',)
#plt.plot(x, acc_list_6, color='purple',)
plt.show()
