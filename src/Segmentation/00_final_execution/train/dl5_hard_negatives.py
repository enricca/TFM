import os
import sys
sys.path.append('/home/alex/lung_cancer_isbi18/src')
import random
import logging
import argparse

import pandas as pd
import numpy as np
from time import time
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from dl_networks.sample_resnet import ResnetBuilder
from dl_model_patches import common
from dl_utils.tb_callback import TensorBoard
from utils import plotting
import matplotlib.pyplot as plt
from skimage import transform

from sklearn.metrics import roc_auc_score

class roc_callback(Callback):
    def __init__(self, X_test, y_test):
        idTrue = np.where(y_test)[0]
        idFalse = np.where(np.logical_not(y_test))[0]
        idSelected = np.concatenate((idTrue[:200], idFalse[:1000]))
        self.x = X_test[idSelected]
        self.y = y_test[idSelected]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        print('\rroc-auc_val: %s\l' %  str(round(roc,4)),  100*' '+'\n')         
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

## PATCHES GENERATION -----------------------------------------------------------------
def compute_ROIs(generate_csv=False, version_dl5 = 0, SCORE_TH = 0.64,
                 patientTxtPath = '/media/shared/datasets/LUNA/CSVFILES/patients_train.txt', 
                 mode = 'train', dataset = 'luna'):
    """Loads the output of DL-IV and load just the 1's (TP or FN's) and the FP's for a given score (SCORE_TH) to train DL-V"""
    if generate_csv:
        with open('/home/shared/output/ROIs_dl5_v{}_{}.csv'.format(version_dl5, mode), 'w') as f:
            f.write('patient,nodules,detected_regions')
    # ## PATIENTS FILE LIST  // not really useful, remove
    if mode == 'train':
        nodules_df = pd.read_csv(OUTPUT_DL4)
    else:
        nodules_df = pd.read_csv(OUTPUT_DL4_TEST)
    nodules_df = nodules_df[(nodules_df['score'] > SCORE_TH) | (nodules_df['label']==1)]

    nodules_df['nslice'] = nodules_df['nslice'].astype(int)
    logging.info("Shape nodules df: %s" % str(nodules_df.shape))

    patients = [ p + '.npz' for p in set(nodules_df['patientid'])]

    files = []
    with open(patientTxtPath, 'r') as f:
        for line in f:
            files.append(line.strip())

    if dataset == 'isbi':
        patients = ['/media/shared/datasets/ISBI/preprocessedNew/' + p for p in patients] 
        filenames = [fp for fp in files if fp in patients]
    else: 
        filenames = [os.path.join(INPUT_PATH, fp) for fp in files if fp in patients]

    def __load_and_store(filename):
        patient_data = np.load(filename)['arr_0'].astype(np.int16)
        patient_data = np.delete(patient_data, 2, 0)
        ndf = nodules_df[nodules_df['patientid']==filename.split('/')[-1].split('.')[0]]
        X, y, rois, stats = common.load_patient(patient_data, ndf, output_rois=True, thickness=1, malignancy=True)
        if not stats:
            stats = {'tp' : 0, 'fn' : 0, 'fp' : 0}

        logging.info("Patient: %s, stats: %s" % (filename.split('/')[-1], stats))
        if generate_csv:
            with open('/home/shared/output/ROIs_dl5_v{}_{}_{}.csv'.format(version_dl5, mode, dataset), 'a') as f:
                f.write('{},{},{}\n'.format(filename.split('/')[-1][:-4], stats['tp'], sum(stats.values())))

        return X, y, stats
    
    common.multiproc_crop_generator(filenames,
                                    os.path.join(PATCHES_PATH,'dl5_v{}_x_{}_{}.npz'.format(version_dl5, mode, dataset)),
                                    os.path.join(PATCHES_PATH,'dl5_v{}_y_{}_{}.npz'.format(version_dl5, mode, dataset)),
                                    __load_and_store)

    

### TRAINING -------------------------------------------------------------------------------------------------------
def chunk_generator(X_orig, y_orig, batch_size=32, augmentation_times=2, thickness=0,
           data_generator = ImageDataGenerator(dim_ordering="th"), is_training=True):
    """
    Batches generator for keras fit_generator. Returns batches of patches 40x40px
     - augmentation_times: number of time to return the data augmented
     - concurrent_patients: number of patients to load at the same time to add diversity
     - thickness: number of slices up and down to add as a channel to the patch
    """
    while 1:
        logging.info("[TRAIN:%s] Loaded batch of patients with %d/%d positives" % (str(is_training), np.sum(y_orig), len(y_orig)))
        #idx_sel = [i for i in range(len(X_orig)) if y_orig[i]==1 or random.uniform(0,1) < 1.2*np.mean(y_orig)]i
        idx_sel  = [i for i in range(len(y_orig)) if y_orig[i]>sigmoid(-1)+0.001 or random.randint(0,9)==0]
        # idx_sel = [i for i in idx_sel if random.randint(0,1,2) == 1]
        X = [X_orig[i] for i in idx_sel]
        y = [y_orig[i] for i in idx_sel]
        logging.info("Downsampled to %d/%d positives" % (np.sum(y), len(y)))
                
       # convert to np array and add extra axis (needed for keras)
        X, y = np.asarray(X), np.asarray(y)
#         y = np.expand_dims(y, axis=1)
        if thickness==0:
            X = np.expand_dims(X, axis=1)
                 
        i, good = 0, 0
        for X_batch, y_batch in data_generator.flow(X, y, batch_size=batch_size, shuffle=is_training):
            i += 1
            if good*batch_size > len(X)*augmentation_times or i>100:  # stop when we have augmented enough the batch
                break
            if X_batch.shape[0] != batch_size:  # ensure correct batch size
                continue
            good += 1
            yield X_batch, y_batch

def sigmoid(y):
    return 1 / (1 + np.exp(1.5-y))

def train(pretrained_model='', version_dl5 = 0, test_dataset='luna'):
    # Data augmentation generator
    # train_datagen = ImageDataGenerator(dim_ordering="th", horizontal_flip=True, vertical_flip=True)
    train_datagen = ImageDataGenerator(
        rotation_range=30, #.06,
        width_shift_range=0.1, #0.02,
        height_shift_range=0.1, #0.02,
        #shear_range=0.0002,
        #zoom_range=0.0002,
        dim_ordering="th",
        horizontal_flip=True,
        vertical_flip=True
        )
    test_datagen = ImageDataGenerator(dim_ordering="th")  # dummy for testing to have the same structure
    
    # LOADING PATCHES FROM DISK
    logging.info("Loading training and test sets")
    x_train = np.load(os.path.join(PATCHES_PATH, 'dl5_v{}_x_train_luna.npz'.format(version_dl5)))['arr_0']
    y_train = np.load(os.path.join(PATCHES_PATH, 'dl5_v{}_y_train_luna.npz'.format(version_dl5)))['arr_0']
    y_train = y_train/84.
    y_train[y_train < 0] = -1
    y_train = sigmoid(y_train)
    y_train = np.expand_dims(y_train, axis=1)

    x_test = np.load(os.path.join(PATCHES_PATH, 'dl5_v{}_x_test_{}.npz'.format(version_dl5, test_dataset)))['arr_0']
    y_test = np.load(os.path.join(PATCHES_PATH, 'dl5_v{}_y_test_{}.npz'.format(version_dl5, test_dataset)))['arr_0']
    y_test = y_test/84.
    y_test[y_test < 0] = -1
    y_test = sigmoid(y_test)
    y_test = np.expand_dims(y_test, axis=1)

    logging.info("Training set (1s/total): %d/%d" % (sum(y_train),len(y_train)))
    logging.info("Test set (1s/total): %d/%d" % (sum(y_test), len(y_test)))
    
    def R2(y_true, y_pred):
        SS_res =  K.sum(K.square(y_true-y_pred)) 
        SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
        return (1-SS_res/(SS_tot + K.epsilon()))
    
    # Load model
    model = ResnetBuilder().build_resnet_50((3,40,40),1)
    model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=[R2,'mse'])
    
    logging.info('Loading exiting model...')
    if pretrained_model != '':
#         model.load_weights(OUTPUT_MODEL)
        model.load_weights(pretrained_model)
        

    model.fit_generator(generator=chunk_generator(x_train, y_train, batch_size=32, thickness=1, data_generator = train_datagen),
                        samples_per_epoch=1280,  # make it small to update TB and CHECKPOINT frequently
                        nb_epoch=500*4,
                        verbose=1,
                        callbacks=[tb, model_checkpoint], # , roc_callback(x_test, y_test)
                        validation_data=chunk_generator(x_test, y_test, batch_size=32, thickness=1, data_generator = test_datagen, is_training=False),
                        nb_val_samples=len(y_test),
                        max_q_size=64,
                        nb_worker=1)  # a locker is needed if increased the number of parallel workers
                 
# ARGUMENTS    
parser = argparse.ArgumentParser(description='Train dl5') 
parser.add_argument('-v_dl4', type = str, required=True, help='version of the DL4 model')
parser.add_argument('-v_dl5', type = str, required=True, help='version of the DL5 model')
parser.add_argument('--train_patches', action='store_true',help='Recompute train patches with txt file?')
parser.add_argument('--test_patches', action = 'store_true', help='Recompute test patches with txt file?')
parser.add_argument('-test_dataset', type = str, default = 'luna', help='Dataset to use for test?(luna, isbi, dsb)')
parser.add_argument('-pretrained_model', default = '', help='path of the pretrained model?')
parser.add_argument('--generate_csv', action = 'store_true', help='generate ROIs csv?')
parser.add_argument('--network', default = 'basic', help='[NOT USED] basic network /hard negatives / malignacy ')

args = parser.parse_args()                 

# PATHS
# wp = os.environ['LUNG_PATH']
wp = '/media/shared/datasets/LUNA/'
od = '/home/shared/'
                 
INPUT_PATH = wp + 'preprocessed_malignancy'
NODULES_PATH = wp + 'CSVFILES/annotations.csv'

OUTPUT_DL4 = od + 'output/luna_dl4_v{}.csv'.format(args.v_dl4)
OUTPUT_DL4_TEST = od + 'output/{}_dl4_v{}.csv'.format(args.test_dataset, args.v_dl4)

PATCHES_PATH = wp + 'patches'

OUTPUT_MODEL = od + 'models/malignancy_hard_negatives_v{}.hdf5'.format(args.v_dl5)
LOGS_PATH = od + 'logs/%s' % 'malignancy_hard_negatives_v{}'.format(args.v_dl5)
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)
                                               
                 
# OTHER INITIALIZATIONS: tensorboard, model checkpoint and logging
tb = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1, write_graph=False, write_images=False)  # replace keras.callbacks.TensorBoard
model_checkpoint = ModelCheckpoint(OUTPUT_MODEL, monitor='val_loss', save_best_only=True)
K.set_image_dim_ordering('th')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')


# CALL FUNCTIONS
if args.train_patches:
    compute_ROIs(args.generate_csv, version_dl5 =args.v_dl5, mode = 'train')
if args.test_patches:
    testpatientsPath = wp + '/CSVFILES/patients_test_isbi.txt' if args.test_dataset == 'isbi' else wp + '/CSVFILES/patients_test.txt'
    compute_ROIs(args.generate_csv, version_dl5 =args.v_dl5, patientTxtPath = testpatientsPath, mode = 'test', dataset = args.test_dataset)

train(args.pretrained_model, args.v_dl5, args.test_dataset)
                 
                 
                 
                 
# Check Nodules Prediction DL4
# nodules_df = pd.read_csv(OUTPUT_DL4)
# th = 0.7
# print 'Avg score NO nodules :', np.mean(nodules_df[nodules_df.label == 0].score)
# print 'Avg score nodules :', np.mean(nodules_df[nodules_df.label == 1].score)
# print 'Pct nodules with score >', th, ':', np.mean(nodules_df[nodules_df.score > th].label)*100, '%'
# print 'Pct nodules with score <=', np.mean(nodules_df[nodules_df.score <= th].label)*100, '%'
# print len(set(nodules_df['patientid']))
# print len(nodules_df['patientid'])

## Example use
# [with patches generation]
# python  dl5_hard_negatives.py -v_dl4=1 -v_dl5=1 --train_patches --test_patches -test_dataset=luna --generate_csv

# [No patches generation] 
# python  am_dl5_hard_negatives.py -v_dl1=1 -v_dl2=1
# python  am_dl5_hard_negatives.py -v_dl1=1 -v_dl2=1 -pretrained_model=jm_patches_hardnegative_v03.hdf5

# python  am_dl5_hard_negatives.py -v_dl1=1 -v_dl2=1 --test_patches -test_dataset=isbi --generate_csv
# python  dl5_hard_negatives.py -v_dl4=1 -v_dl5=1 -test_dataset=luna --generate_csv
