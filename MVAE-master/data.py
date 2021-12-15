import numpy as np
a =np.loadtxt('shuju.txt')
print(a.shape)
a=a.T
print(a.shape)
np.savetxt("data.txt", a, fmt="%f", delimiter=",")