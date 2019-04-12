import matplotlib.pyplot as plt
import numpy as np
import torch
filename = 'loss_final.npy'
data = np.load(filename)
data = data[0:-20000:150]
print(data.shape)
plt.plot(data,label='loss')
plt.xlabel('iter_n')
plt.ylabel('loss')
plt.ylim((-2.15, -1))
# plt.ylim((1,2.23))
plt.legend()
plt.show()