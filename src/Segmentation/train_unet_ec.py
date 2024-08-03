import sys
sys.path.append('/homedtic/ecosp/Code')
sys.path.append('/homedtic/ecosp/lung_cancer_isbi18/src')
import numpy as np
import dl_networks.unet2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import skimage, skimage.measure
from memory_profiler import profile
import pandas
import dl_networks.unet3

X= np.load( '/homedtic/ecosp/patches/unet_v1_train_x.npz')['arr_0']        # patches from mask
y = np.load('/homedtic/ecosp/patches/unet_v1_train_y.npz')['arr_0']        # mask de cada patch

data = {'x_test': X[-300:],
        'y_test' : y[-300:],
        'x_train': X[:-300],
        'y_train' : y[:-300]}

#data_swapped = { k : np.swapaxes(v, 1, 3) for k, v in data.iteritems()}
data_swapped = data
# les mascares que treiem són 32x32 (les imatges d'entrada són 96x96 pel padding)
data_swapped['y_train'] = data_swapped['y_train'][:,:, 32:64, 32:64]
data_swapped['y_test'] = data_swapped['y_test'][:,:, 32:64, 32:64]

# now we will use the patches to train the unet3 network
reload(dl_networks.unet3)

# create UNET and train (save weights)
u = dl_networks.unet3.UNet(ordering= 'channels_first')
#model = u.create_model([ 3 , 32 *40, 32*40])
model = u.create_model([ 3, 96, 96 ])

output = '/homedtic/ecosp/unet_ec_v1.hdf5'
u.train_model(model, data_swapped, LOGS_PATH = '/homedtic/ecosp/logs/', OUTPUT_MODEL = output)
