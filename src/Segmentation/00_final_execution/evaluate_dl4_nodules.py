"""
Run the network over the testset.

Uses load_patient to generate patches over, and for each patch gets the probability that it contains a nodule.
"""

import sys
import os
sys.path.append('/home/alex/lung_cancer_isbi18/src')

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

# python evaluate_dl4_nodules.py -input_model /home/shared/models/malignancy_nodules_v1.hdf5 -input_data /media/shared/datasets/ISBI/preprocessed_main_nodule_mask -output_csv /home/shared/output/isbi_dl4_nodules_v1.csv -dataset isbi --overwrite

# python evaluate_dl4_nodules.py -input_model /home/shared/models/malignancy_nodules_v1.hdf5 -input_data /media/shared/datasets/LUNA/preprocessed_extra_features -output_csv /home/shared/output/luna_dl4_nodules_v1.csv -dataset luna --overwrite

# python evaluate_dl4_nodules.py -input_model /home/shared/models/malignancy_nodules_v3.hdf5 -input_data /media/shared/datasets/DSB/preprocessed_s1 -output_csv /home/shared/output/dsb_dl4_nodules_v3.csv -dataset dsb --overwrite

def sigmoid(y):
    return 1 / (1 + np.exp(1.5-y))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs a trained in a preprocessed dataset (npz)')
    parser.add_argument('-input_model', help='path of the model')
    parser.add_argument('-input_data', help='path of the input data')
    parser.add_argument('-output_csv', help='path of the output csv')   
    parser.add_argument('-dataset', help='Eval on isbi, luna, dsb?')    
    parser.add_argument('--roi_statistics_csv', default = '', help=' (OPTIONAL) Annotate statistics')
    parser.add_argument('--overwrite',  action='store_true', help=' (OPTIONAL) Overwrite Default none.')
    parser.add_argument('--convertToFloat',  action='store_true', help=' (OPTIONAL) Transform the images to float. Dunno why, but some networks only work with one kind (Mingot ones with float, new ones with int16).')
    
    args = parser.parse_args()
      
    #Load the network
    K.set_image_dim_ordering('th')
    model = ResnetBuilder().build_resnet_50((3,40,40),1)
    model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['mse'])
    logging.info('Loading existing model %s...' % args.input_model)
    model.load_weights(args.input_model)
   
    # Get files
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

    if args.dataset == 'dsb': # Hard coded for now to read anotated nodules. 
        nodules_df = '/home/shared/output/dsb_dl2_v1_aggregated_manualFilter_v6.csv'
        nodules_df = pd.read_csv(nodules_df)
        nodules_df['nsliceFrom'] = (nodules_df['nslice'] - (nodules_df['nslicesSpread'] - 1)/2).astype(int)
        nodules_df['nsliceTo'] = (nodules_df['nslice'] + (nodules_df['nslicesSpread'] - 1)/2).astype(int)

    f = open(args.output_csv, 'w')
    f.write('patientid,nslice,x,y,diameter,score\n')
    
    #Set threshold
    nPatients = len(patientFiles)
    tStart = time.time()
    no_nodules = 0

    for pN, patientPath in enumerate(patientFiles):
        try:
            patientName = patientPath.split('/')[-1][:-4]
            patient_data = np.load(patientPath)['arr_0'].astype(np.int16)
            # put feature at channel 3 and delete the rest
            print (patient_data.shape)
            if args.dataset == 'isbi':
                patient_data = np.delete(patient_data, 2, 0) # only for isbi, we delete nodules mask to use the selected-nodule mask
            if args.dataset == 'luna':
                patient_data = np.delete(patient_data, 5, 0) # only for luna, we delete spi mask
                patient_data = np.delete(patient_data, 4, 0) # only for luna, we delete lob mask
                patient_data = np.delete(patient_data, 3, 0) # only for luna, we delete mal mask
            logging.info("Patient %s read." % patientName)
            
            if args.dataset == 'dsb': # Hard coded for now to read anotated nodules. 
                ndf = nodules_df[nodules_df['patientid']==patientPath.split('/')[-1].split('.')[0]]
                if len(ndf) == 0:
                    print('Error. No nodules anotated!')
                    print(patientName)
                    no_nodules += 1
                    continue
                
                extra_slices_df = pd.DataFrame()
                for idx, row in ndf.iterrows():
                    for sl in range(row['nsliceFrom'], row['nsliceTo'] + 1):
                        aux = row.copy()
                        aux['nslice'] = sl
                        extra_slices_df = extra_slices_df.append(aux)  
                extra_slices_df['nslice'] = extra_slices_df['nslice'].astype(np.int16)
                X, y, rois, stats = common.load_patient(patient_data, extra_slices_df, discard_empty_nodules=True, 
                                                                     output_rois=True, thickness=1)
            else:
                X, y, rois, stats = common.load_patient(patient_data, output_rois=True, thickness=1, feature = True, 
                                                        nodules_as_rois = 1, discard_empty_nodules = True, 
                                                        include_ground_truth=True, ground_truth_only = True)

            print(stats)
            ## True Nodules
            nod_i = [i for i in range(len(y)) if y[i] == 1]
            if args.dataset != 'dsb':
                print('Identified ' + str(len(nod_i)) + ' nodules-slice')

                if len(nod_i) == 0:
                    print('Error. No nodules!')
                    print(patientName)
                    no_nodules += 1
                    continue

            X = np.asarray(X)
            preds = model.predict(X, verbose=2)
                    
            for i, p in enumerate(preds):
                if y[i] != 1 and args.dataset != 'dsb': 
                    continue
                nslice, r = rois[i]
                f.write('%s,%d,%d,%d,%.3f,%.5f\n' % (patientName, nslice, r.centroid[0], r.centroid[1], r.equivalent_diameter,preds[i]))

            logging.info("Predicted patient %s (%d/ %d). Batch results: %d/%d, Max pred %f" % \
                         ( patientPath, pN, nPatients, len([p for p in preds if p>0.7]),len(preds), max(preds)) )
            if args.roi_statistics_csv:
                logging.info("Patient: %s, stats ROI: %s" % (patientName, statsROIGeneration))
                statsROIGeneration['pId'] = patientName
                stats_roi_pd = stats_roi_pd.append(statsROIGeneration, ignore_index = True)

            timeFromStart = time.time() - tStart
            timePerPatientMean = timeFromStart / (pN + 1)
            timeRemainingAprox = timePerPatientMean * (nPatients - 1 - pN)
            logging.info("Time elapsed %s  / Remaining %s" % 
                         (str(datetime.timedelta(seconds=timeFromStart)), str(datetime.timedelta(seconds=timeRemainingAprox)))
                        )
        except Exception as e:
            logging.error("Error processing patient %s, skipping. %s" % (patientName, str(e)))
    print('{nerrs} nodules not find!'.format(nerrs = no_nodules))      
    if args.roi_statistics_csv:
        stats_roi_pd.to_csv(args.roi_statistics_csv)
                       

