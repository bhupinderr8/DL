import numpy as np
import scipy.io as sio


X_data = np.load('X_data.npy')
test = sio.loadmat('anno.mat')

Y_data = []

for i in range(9344):
    if test['anno']['y'][0][0][1][i] == 1:
        Y_data.append(test['anno']['y'][0][0][0][i])

Y = np.asarray(Y_data)
for i in range(7048):
    if(Y_data[i]==-1):
        Y[i] = 0
    else:
        Y[i] = 1

np.save('Y_data_not_ont_hot.npy', Y)