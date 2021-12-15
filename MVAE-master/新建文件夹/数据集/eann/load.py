import pickle
import numpy as np
f1 = open('vectors.txt', 'rb')
vectors = pickle.load(f1)
f1.close()
f1 = open('labels.txt', 'rb')
labels = pickle.load(f1)
f1.close()
print(vectors)
print(labels)
vectors = np.array(vectors)
labels = np.array(labels)
np.savetxt("1.txt", vectors, fmt="%f", delimiter=" ")
print(labels[:, 0])
labels=labels[:, 0]
labels=labels.T
print(labels.shape)
np.savetxt("1_label.txt", labels, fmt="%d", delimiter=" ")