"""
Run the network over the testset.

Uses load_patient to generate patches over, and for each patch gets the probability that it contains a nodule.
"""

import sys
import os
import logging
import argparse
import numpy as np
import pandas as pd
import time,datetime
from dl_model_patches import  common
import keras
from keras import backend as K
from dl_networks.sample_resnet import ResnetBuilder
from keras.optimizers import Adam

# python evaluate_dl4.py -input_model /home/shared/models/malignancy_v1.hdf5 -input_data /media/shared/datasets/LUNA/preprocessed_malignancy/ -output_csv /home/shared/output/luna_dl4_v1.csv
# python evaluate_dl4.py -input_model /home/shared/models/malignancy_v1.hdf5 -input_data /media/shared/datasets/ISBI/preprocessedNew/ -output_csv /home/shared/output/isbi_dl4_v1.csv
def sigmoid(y):
    return 1 / (1 + np.exp(1.5-y))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs a trained in a preprocessed dataset (npz)')
    parser.add_argument('-input_model', help='path of the model')
    parser.add_argument('-input_data', help='path of the input data')
    parser.add_argument('-output_csv', help='path of the output csv')
    parser.add_argument('--roi_statistics_csv', default = '', help=' (OPTIONAL) Annotate statistics')
    parser.add_argument('--threshold', type = float, default = -1, help=' (OPTIONAL) Discard patches with less than that.')
    parser.add_argument('--overwrite',  action='store_true', help=' (OPTIONAL) Overwrite Default none.')
    parser.add_argument('--convertToFloat',  action='store_true', help=' (OPTIONAL) Transform the images to float. Dunno why, but some networks only work with one kind (old ones with float, new ones with int16).')
    args = parser.parse_args()

    #Load the network
    K.set_image_dim_ordering('th')
    model = ResnetBuilder().build_resnet_50((3,40,40),1)
    model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['mse'])
    logging.info('Loading existing model %s...' % args.input_model)
    model.load_weights(args.input_model)
    
    #Create a dataframe for the ROIS
    stats_roi_pd = pd.DataFrame()
    
    #Get the patient files
    if os.path.isdir(args.input_data):
        patientFiles = map(lambda s: os.path.join(args.input_data, s)  ,filter(lambda s: s.endswith('.npz'), os.listdir(args.input_data)))
    else:
        patientFiles = []
        with open(args.input_data, 'r') as f:
            for line in f:
                patientFiles.append(line.strip())

    #Set Output
    if not args.overwrite and os.path.exists(args.output_csv):
        logging.warning('Output path already exists! After the indicident, you need to turn the overwrite option on. Computing only patients that do not appear in the csv.')
        df = pd.read_csv(args.output_csv)
        patientFiles = filter(lambda s: s not in df.patientid, patientFiles)
        
    f = open(args.output_csv, 'w')
    f.write('patientid,nslice,x,y,diameter,score,label\n')
    
    #Set threshold
    threshold = -1 #deactivated
    nPatients = len(patientFiles)
    tStart = time.time()
    for pN, patientPath in enumerate(patientFiles):
        try:
            patientName = patientPath.split('/')[-1][:-4]
            patient_data = np.load(patientPath)['arr_0'].astype(np.int16)
            # put feature at channel 3 and delete the rest
            patient_data = np.delete(patient_data, 2, 0) # only for isbi and luna
            logging.info("Patient %s read." % patientName)

            X, y, rois, statsROIGeneration = common.load_patient(patient_data, discard_empty_nodules=False, output_rois=True, thickness=1, convertToFloat = args.convertToFloat, feature=True)
            logging.info("Patient %s ROIS extracted." % patientName)
            y = np.array(y)
            # float features are multiplied by 84, normalize them again
            y = y/84.
            y[y < 0] = -1
            y = sigmoid(y)
            y = list(y)
            X = np.asarray(X)
            preds = model.predict(X, verbose=2)

            logging.info("Predicted patient %s (%d/ %d). Batch results: %d/%d (th=0.7), Max pred %f" % \
                         ( patientPath, pN, nPatients, len([p for p in preds if p>0.7]),len(preds), max(preds)) )
            if args.roi_statistics_csv:
                logging.info("Patient: %s, stats ROI: %s" % (patientName, statsROIGeneration))
                statsROIGeneration['pId'] = patientName
                stats_roi_pd = stats_roi_pd.append(statsROIGeneration, ignore_index = True)
            for i, p in enumerate(preds):
                if p < threshold:
                    continue
                nslice, r = rois[i]
                f.write('%s,%d,%d,%d,%.3f,%.5f,%.5f\n' % (patientName, nslice, r.centroid[0], r.centroid[1], r.equivalent_diameter,preds[i],y[i]))
            timeFromStart = time.time() - tStart
            timePerPatientMean = timeFromStart / (pN + 1)
            timeRemainingAprox = timePerPatientMean * (nPatients - 1 - pN)
            logging.info("Time elapsed %s  / Remaining %s" % 
                         (str(datetime.timedelta(seconds=timeFromStart)), str(datetime.timedelta(seconds=timeRemainingAprox)))
                        )
        except Exception as e:
            logging.error("Error processing patient %s, skipping. %s" % (patientName, str(e)))
            
    if args.roi_statistics_csv:
        stats_roi_pd.to_csv(args.roi_statistics_csv)
                       

