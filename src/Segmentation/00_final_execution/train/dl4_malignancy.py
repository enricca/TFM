import sys
import getpass
import os
import random
import logging
import argparse

import numpy as np
import pandas as pd
from dl_model_patches import  common

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
#from dl_networks.sample_resnet import ResnetBuilder

from dl_utils.tb_callback import TensorBoard
from dl_networks.sample_resnet import ResnetBuilder
from sklearn.metrics import roc_auc_score

# class roc_callback(Callback):
#     def __init__(self, X_test, y_test):
#         idTrue = np.where(y_test)[0]
#         idFalse = np.where(np.logical_not(y_test))[0]
#         idSelected = np.concatenate((idTrue[:200], idFalse[:1000]))
#         self.x = X_test[idSelected]
#         self.y = y_test[idSelected]

#     def on_train_begin(self, logs={}):
#         return

#     def on_train_end(self, logs={}):
#         return

#     def on_epoch_begin(self, epoch, logs={}):
#         return

#     def on_epoch_end(self, epoch, logs={}):
#         y_pred = self.model.predict(self.x)
#         roc = roc_auc_score(self.y, y_pred)
#         print('\rroc-auc_val: %s\l' %  str(round(roc,4)),  100*' '+'\n')         
#         return

#     def on_batch_begin(self, batch, logs={}):
#         return

#     def on_batch_end(self, batch, logs={}):
#         return
    
## PATCHES GENERATION -----------------------------------------------------------------
def compute_ROIs(generate_csv=False, 
                 version = 0,
                 patientTxtPath = '/media/shared/datasets/LUNA/CSVFILES/patients_train.txt',
                 mode = 'train'):
    if generate_csv:
        with open(os.path.join(OUTPUT_CSV,'ROIs_dl4_v{}_{}.csv'.format(version, mode)), 'w') as f:
            f.write('patient,nodules,detected_regions')
    # ## PATIENTS FILE LIST  // not really useful, remove
    filter_annotated = False
    patients_with_annotations = pd.read_csv(NODULES_PATH)  # filter patients with no annotations to avoid having to read them
    patients_with_annotations = list(set(patients_with_annotations['seriesuid']))
    patients_with_annotations = ["luna_%s.npz" % p.split('.')[-1] for p in patients_with_annotations]
    
    filenames = []
    with open(patientTxtPath, 'r') as f:
        for line in f:
            filenames.append(line.strip())
    filenames_train = [os.path.join(INPUT_PATH, fp) for fp in filenames if fp in patients_with_annotations or not filter_annotated]

    def __load_and_store(filename):
        patient_data = np.load(filename)['arr_0']
        patient_data = patient_data.astype(np.int16)
        patient_data = np.delete(patient_data, 2, 0)
        X, y, rois, stats = common.load_patient(patient_data, discard_empty_nodules=True, output_rois=True, debug=True, include_ground_truth=True, thickness=1, feature=True)
        if not stats: 
            stats = {'tp' : 0, 'fn' : 0, 'fp' : 0}

        logging.info("Patient: %s, stats: %s" % (filename.split('/')[-1], stats))
        if generate_csv:
            with open(os.path.join(OUTPUT_CSV,'ROIs_dl4_v{}_{}.csv'.format(version, mode)), 'a') as f:
                f.write('{},{},{}\n'.format(filename.split('/')[-1][:-4], stats['tp'], sum(stats.values())))
        # If the patient is empty
        return X, y, stats



    common.multiproc_crop_generator(filenames_train,
                                    os.path.join(PATCHES_PATH, 'dl4_v{}_x_{}.npz'.format(version, mode)),
                                    os.path.join(PATCHES_PATH, 'dl4_v{}_y_{}.npz'.format(version, mode)),
                                    __load_and_store)


### TRAINING -----------------------------------------------------------------
def chunks(X, y, batch_size=32, augmentation_times=4, thickness=0,
           data_generator = ImageDataGenerator(dim_ordering="th"), is_training=True):
    """
    Batches generator for keras fit_generator. Returns batches of patches 40x40px
     - augmentation_times: number of time to return the data augmented
     - concurrent_patients: number of patients to load at the same time to add diversity
     - thickness: number of slices up and down to add as a channel to the patch
    """
    while 1:
        prct_pop, prct1 = 0.2, 0.2  # (1) of all the training set, how much we keep (2) % of 1's
        #idx_1 = [i for i in range(len(y)) if y[i]==1]
        idx_1 = [i for i in range(len(y)) if y[i]>sigmoid(-1)+0.001] # for malignancy

        idx_1 = random.sample(idx_1, int(prct_pop*len(idx_1)))
        idx_0 = [i for i in range(len(y)) if y[i]<=sigmoid(-1)+0.001] # for malignancy
        idx_0 = random.sample(idx_0, int(len(idx_1)/prct1))
        selected_samples = idx_0 + idx_1
        random.shuffle(selected_samples)
        logging.info("Final downsampled dataset stats: TP:%d, FP:%d" % (sum(y[selected_samples]>0), len(y[selected_samples])-sum(y[selected_samples]>0)))
        i, good = 0, 0
        lenX = len(selected_samples)
        
        for X_batch, y_batch in data_generator.flow(X[selected_samples], y[selected_samples], batch_size=batch_size, shuffle=is_training):
            i += 1
            if good*batch_size > lenX*augmentation_times or i>100:  # stop when we have augmented enough the batch
                break
            if X_batch.shape[0] != batch_size:  # ensure correct batch size
                continue
            good += 1
            yield X_batch, y_batch

def sigmoid(y):
    return 1 / (1 + np.exp(1.5-y))
    
    
def train(load_model=False, version = 0):
    # Data augmentation generator
    train_datagen = ImageDataGenerator(
        rotation_range=30,  # .06,
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
    x_test = np.load(os.path.join(PATCHES_PATH, 'dl4_v{}_x_test.npz'.format(version)))['arr_0']
    y_test = np.load(os.path.join(PATCHES_PATH, 'dl4_v{}_y_test.npz'.format(version)))['arr_0']
    y_test = y_test/84.
    y_test[y_test < 0] = -1
    y_test = sigmoid(y_test)
    y_test = np.expand_dims(y_test, axis=1)

    
    x_train = np.load(os.path.join(PATCHES_PATH, 'dl4_v{}_x_train.npz'.format(version)))['arr_0']
    y_train = np.load(os.path.join(PATCHES_PATH, 'dl4_v{}_y_train.npz'.format(version)))['arr_0']
    y_train = y_train/84.
    y_train[y_train < 0] = -1
    y_train = sigmoid(y_train)
    y_train = np.expand_dims(y_train, axis=1)
#     logging.info("Training set (1s/total): %d/%d" % (sum(y_train),len(y_train)))
#     logging.info("Test set (1s/total): %d/%d" % (sum(y_test), len(y_test)))

    # Load model
    def R2(y_true, y_pred):
        SS_res =  K.sum(K.square(y_true-y_pred)) 
        SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
        return (1-SS_res/(SS_tot + K.epsilon()))
    
    model = ResnetBuilder().build_resnet_50((3,40,40),1)
    model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=[R2, 'mse']) 
    if load_model:
        logging.info('Loading exiting model...')
        model.load_weights(OUTPUT_MODEL)


    model.fit_generator(generator=chunks(x_train, y_train, batch_size=32, thickness=1, data_generator = train_datagen),
                        samples_per_epoch=1280,  # make it small to update TB and CHECKPOINT frequently
                        nb_epoch=1600,
                        verbose=1,
                        #class_weight={0:1., 1:4.},
                        callbacks=[tb, model_checkpoint], # roc_callback(x_test, y_test)],
                        validation_data=chunks(x_test, y_test, batch_size=32, thickness=1, data_generator = test_datagen, is_training=False,),
                        nb_val_samples=32*40,
                        max_q_size=10,
                        # initial_epoch=715,
                        nb_worker=1)  # a locker is needed if increased the number of 

    
# ARGUMENTS    
parser = argparse.ArgumentParser(description='Train dl4')
parser.add_argument('-version', type = str, required=True, help='version of the model')
parser.add_argument('-train', type = str, 
                    default ='', 
                    help='Recompute train patches with txt  file with the paths of the train npz.')
parser.add_argument('-test', type = str,  default = '',
                    help='Recompute train patches with the patients in the file.')
parser.add_argument('--load_model', action = 'store_true',  help='load pretrained model?')
parser.add_argument('--generate_csv', action = 'store_true',  help='generate ROIs csv?')
parser.add_argument('--network', default = 'basic', help='[NOT USED] basic network /hard negatives / malignacy ')

args = parser.parse_args()


# PATHS
# wp = os.environ['LUNG_PATH']
wp = '/media/shared/datasets/LUNA/'
od = '/home/shared/'
INPUT_PATH = wp + 'preprocessed_malignancy'  # INPUT_PATH = wp + 'data/preprocessed5_sample'
# VALIDATION_PATH = wp + 'preprocessed_validation_luna'
NODULES_PATH = wp + 'CSVFILES/annotations.csv'
PATCHES_PATH = wp + 'patches'  # PATCHES_PATH = wp + 'data/preprocessed5_patches'

OUTPUT_MODEL = od + 'models/malignancy_v{}.hdf5'.format(args.version)  # OUTPUT_MODEL = wp + 'personal/jm_patches_train_v06_local.hdf5'
LOGS_PATH = od + 'logs/%s' % 'malignancy_v{}'.format(args.version)
OUTPUT_CSV = od + 'output/'
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)

# OTHER INITIALIZATIONS: tensorboard, model checkpoint and logging, auc?
tb = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1, write_graph=False, write_images=False)  # replace keras.callbacks.TensorBoard
model_checkpoint = ModelCheckpoint(OUTPUT_MODEL, monitor='val_loss', save_best_only=True)
K.set_image_dim_ordering('th')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')


# CALL FUNCTIONS
if args.train:
    compute_ROIs(args.generate_csv, version =args.version, patientTxtPath = args.train,  mode = 'train')
if args.test:
    compute_ROIs(args.generate_csv, version =args.version, patientTxtPath = args.test, mode = 'test')

train(args.load_model, args.version)


    
