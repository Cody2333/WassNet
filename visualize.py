import matplotlib.pyplot as plt
import numpy as np
import torch
d=torch.load('w_dist_99.npy', map_location='cpu')
print(d.shape)
data = np.load('w_dist_99.npy')
print(data.shape)
plt.plot(data[100:],label='loss')
plt.legend()
plt.show()