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
import tensorboard as tf

X= np.load( '/homedtic/ecosp/patches/unet_v1_train_x.npz')['arr_0']        # patches from mask
y = np.load('/homedtic/ecosp/patches/unet_v1_train_y.npz')['arr_0']        # mask de cada patch
#X= np.load( '/home/enric/Desktop/TFM/DATA_LUNA/patches/unet_v1_train_x.npz')['arr_0']        # patches from mask
#y = np.load('/home/enric/Desktop/TFM/DATA_LUNA/patches/unet_v1_train_y.npz')['arr_0']        # mask de cada patch


"""data = {'x_test': X[-300:],
        'y_test' : y[-300:],
        'x_train': X[-400:-300],
        'y_train' : y[-400:-300]}"""

data = {'x_test': X[-300:],
        'y_test' : y[-300:],
        'x_train': X[:-300],
        'y_train' : y[:-300]}

#data_swapped = { k : np.swapaxes(v, 1, 3) for k, v in data.iteritems()}
data_swapped = data


# now we will use the patches to train the unet3 network
import dl_networks.unet3
reload(dl_networks.unet3)

# create UNET and train (save weights)
u = dl_networks.unet3.UNet(ordering= 'channels_first')
#model = u.create_model([ 3 , 32 *40, 32*40])
model = u.create_model([3, 96, 96 ])
#model = u.create_model([ 3 ])
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

#u.train_model(model, data_swapped, LOGS_PATH = '/homedtic/ecosp/logs/', OUTPUT_MODEL = '/homedtic/ecosp/unet_enric.hdf5')
#u.train_model(model, data_swapped, LOGS_PATH = '/home/enric/Desktop/TFM/DATA_LUNA/logs/', OUTPUT_MODEL = '/home/enric/Desktop/TFM/DATA_LUNA/unet_enric.hdf5')

#model.load_weights('/homedtic/ecosp/unet_enric.hdf5')
#model.load_weights('/home/enric/Desktop/TFM/models/unet_enric.hdf5')
#model.load_weights('/homedtic/ecosp/unet_pesosGabriel.hdf5')
model.load_weights('/homedtic/ecosp/unet.hdf5')

l = len( data['x_test'][:,1,1,1] )
ground_truth = data['y_test']
# pillar nomes el centre de cada data['y_test']
ground_truth = ground_truth[ : , :, 32:64, 32:64 ]

#output_csv = 'predictions_ec.csv'
#f = open(output_csv, 'w')
#f.write('prediction', 'ground_truth\n')
print('apunt de loop')

# image data as uint16
X = data['x_test']
X = X.astype('uint16')


threshold = -1
preds = model.predict(X, verbose=2)
#print(preds)

for i, x in enumerate( preds ):
    for j, y in enumerate( preds[ i,: ,: ,: ] ):
        for k,z in enumerate( preds[i, :, j, : ] ):
            if( preds[ i ,: ,j, k ] > 0.9 ):
                preds[i ,: ,j, k ] = 1
            else:
                preds[i ,: ,j, k ] = 0



np.savez('prediction', preds)

print( 'preds: ', preds.shape )
#print(preds.type)   # numpy ndarray
print( 'ground truth: ', ground_truth.shape )

#from scipy.spatial import distance
#sc = distance.dice( ground_truth, preds )
from dice_warner import dice
sc = dice(ground_truth, preds)
print(sc)


tf.summary.histogram("predictions", preds)


#print('auc: ', metrics.roc_auc_score(ground_truth, preds ) )

i = 150

#plt.figure(figsize = (10, 10))
#plt.imshow(X[i, 1, :, :], cmap = 'gray')
#plt.contour(preds[i, 0, :, :] < 0.5)
#plt.contour(ground_truth[i, 0, :, :] , colors = 'red')

print('finiiiiiiiished')










