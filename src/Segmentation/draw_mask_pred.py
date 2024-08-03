import sys
sys.path.append('/homedtic/ecosp/Code')
sys.path.append('/homedtic/ecosp/lung_cancer_isbi18/src')
import numpy as np
import dl_networks.unet2
import numpy as np
import os
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import skimage, skimage.measure
from sklearn import metrics
import h5py
from memory_profiler import profile
import pandas

X= np.load( 'patches/unet_v1_train_x.npz')['arr_0']        # patches from mask
y = np.load('patches/unet_v1_train_y.npz')['arr_0']        # mask de cada patch

data = {'x_test': X[-300:],
        'y_test' : y[-300:],
        'x_train': X[:-300],
        'y_train' : y[:-300]}

ground_truth = data['y_test']
ground_truth = ground_truth[ : , :, 32:64, 32:64 ]

preds = np.load('patches/prediction.npz')

i = 3
preds[i, 0, :, :]

plt.figure(figsize = (10, 10))
plt.imshow(X[i, 1, :, :], cmap = 'gray')
plt.contour(preds[i, 0, :, :] > 0.5, colors = 'yellow')
plt.contour(ground_truth[i, 0, :, :] , colors = 'red')

preds.shape
aux = preds[43, 0, :, :]
for i in range(30):
        aux = preds[i, 0, :, :]
        if( aux.any() == 1 ):
                print(i)
        elif( aux.all() == 0 ):
                print(i, '= 0')
