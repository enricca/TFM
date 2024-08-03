"""
Run the network over the testset.

Uses load_patient to generate patches over, and for each patch gets the probability that it contains a nodule.
"""

import sys
import os
sys.path.insert(0, 'lung_cancer_isbi18/src/')
import logging
import argparse
import numpy as np
import pandas as pd
from time import time
import datetime
from dl_model_patches import common

import keras
from keras import backend as K
from dl_networks.sample_resnet import ResnetBuilder
from keras.optimizers import Adam


def ts():
    return datetime.datetime.now().strftime('%y-%m-%d %H:%M:%S')

def sigmoid(y):
    return 1 / (1 + np.exp(1.5-y))

def transform_y(y):
    y = np.array(y)
    y = y/84.
    y[y < 0] = -1
    y = sigmoid(y)
    y = list(y)
    return y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs a trained in a preprocessed dataset (npz)')
    parser.add_argument('-dl1', help='path of the model NODULARITY')
    parser.add_argument('-dl2', help='path of the model')
    #parser.add_argument('-dl3', help='path of the model')
    parser.add_argument('-dl4', help='path of the model MALIGNANCY')
    parser.add_argument('-dl5', help='path of the model')
    parser.add_argument('-dl6', help='path of the model LOBULATION')
    parser.add_argument('-dl7', help='path of the model')
    parser.add_argument('-dl8', help='path of the model SPICULATION')
    parser.add_argument('-input_data', help='path of the input data')
    parser.add_argument('-output_csv', help='path of the output csv')
    parser.add_argument('--roi_statistics_csv', default = '', help=' (OPTIONAL) Annotate statistics')
    parser.add_argument('--threshold', type = float, default = -1, help=' (OPTIONAL) Discard patches with less than that.')
    parser.add_argument('--overwrite',  action='store_true', help=' (OPTIONAL) Overwrite Default none.')
    parser.add_argument('--convertToFloat',  action='store_true', help=' (OPTIONAL) Transform the images to float. Dunno why, but some networks only work with one kind (Mingot ones with float, new ones with int16).')

    args = parser.parse_args()

    #Load the networks
    K.set_image_dim_ordering('th')

    if args.dl1:
        model_dl1 = ResnetBuilder().build_resnet_50((3,40,40),1)
        model_dl1.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy','fmeasure'])
        logging.info('Loading nodularity model %s...' % args.dl1)
        model_dl1.load_weights(args.dl1)

    if args.dl4:
        model_dl4 = ResnetBuilder().build_resnet_50((3,40,40),1)
        model_dl4.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['mse'])
        logging.info('Loading malignancy model %s...' % args.dl4)
        model_dl4.load_weights(args.dl4)

    if args.dl6:
	    model_dl6 = ResnetBuilder().build_resnet_50((3,40,40),1)
        model_dl6.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['mse'])
        logging.info('Loading lobulation model %s...' % args.dl6)
        model_dl6.load_weights(args.dl6)

    if args.dl8:
	    model_dl8 = ResnetBuilder().build_resnet_50((3,40,40),1)
        model_dl8.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['mse'])
        logging.info('Loading spiculation model %s...' % args.dl8)
        model.load_weights(args.dl8)


    #Create a dataframe for the ROIS
    stats_roi_pd = pd.DataFrame()

    #Get the patient files
    patientFiles = list(map(lambda s: os.path.join(args.input_data, s)  ,filter(lambda s: s.endswith('.npz'), os.listdir(args.input_data))))

    #Set Output
    if not args.overwrite and os.path.exists(args.output_csv):
        raise Exception('Output path already exists! After the indicident, you need to turn the overwrite option on.')


    csv_headers = ['patientid','nslice','x','y','diameter']

    if args.dl1:
        csv_headers.extend(['score_nodule', 'label_nodule'])
    if args.dl4:
        csv_headers.extend(['score_malignancy', 'label_malignancy'])
    if args.dl6:
        csv_headers.extend(['score_lobulation', 'label_lobulation'])
    if args.dl8:
        csv_headers.extend(['score_spiculation', 'label_spiculation'])


    dataset = args.input_data.split('/')[-2]
    f = open(args.output_csv, 'w')
    f.write(','.join(csv_headers)+'\n')


    #Set threshold
    threshold = -1 #deacti ated
    nPatients = len(patientFiles)

    print('Starting loop over patients...')
    for pN, patientPath in enumerate(patientFiles[:10]):
        try:
            patientName = patientPath.split('/')[-1][:-4]
            patient_data = np.load(patientPath)['arr_0'].astype(np.int16)
    	    print('{} - Loading patient {}...'.format(ts(), patientName))
            X, y_nodules, rois, statsROIGeneration = common.load_patient(patient_data[[0,1,2]], discard_empty_nodules=False, output_rois=True, thickness=1)
    	    X = np.asarray(X)
    	    preds_nodules = model_dl1.predict(X, verbose=2)

            if args.dl4:
        		X, y_malignancy, rois, statsROIGeneration = common.load_patient(patient_data[[0,1,3]], discard_empty_nodules=False, output_rois=True, thickness=1, convertToFloat = args.convertToFloat, feature=True)
        		y_malignancy = transform_y(y_malignancy)
        		X = np.asarray(X)
        		preds_malignancy = model_dl4.predict(X, verbose=2)
            if args.dl6:
                X, y_lobulation, rois, statsROIGeneration = common.load_patient(patient_data[[0,1,4]], discard_empty_nodules=False, output_rois=True, thickness=1, convertToFloat = args.convertToFloat, feature=True)
        		y_lobulation = transform_y(y_lobulation)
        		X = np.asarray(X)
        		preds_lobulation = model_dl6.predict(X, verbose=2)
            if args.dl8:
                X, y_spiculation, rois, statsROIGeneration = common.load_patient(patient_data[[0,1,5]], discard_empty_nodules=False, output_rois=True, thickness=1, convertToFloat = args.convertToFloat, feature=True)
        		y_spiculation = transform_y(y_spiculation)
        		X = np.asarray(X)
        		preds_spiculation = model_dl8.predict(X, verbose=2)


                logging.info("Predicted patient %s (%d/ %d). Batch results: %d/%d (th=0.7)" % \
                             ( patientPath, pN, nPatients, len([p for p in preds_nodules if p>0.7]),len(preds_nodules)))
                if args.roi_statistics_csv:
                    logging.info("Patient: %s, stats ROI: %s" % (patientName, statsROIGeneration))
                    statsROIGeneration['pId'] = patientName
                    stats_roi_pd = stats_roi_pd.append(statsROIGeneration, ignore_index = True)
                #print('Starting loop over nodules...')
    	    for i, p in enumerate(preds_nodules):
                    if p < threshold:
                        continue
                    nslice, r = rois[i]

                    nodule = [patientName, str(nslice), str(r.centroid[0]), str(r.centroid[1]), str(r.equivalent_diameter)]
                    if args.dl1:
                        nodule.append(str(preds_nodules[i][0]))
                        if dataset == 'LUNA':
                            nodule.append(str(y_nodules[i]))
                        else:
                            nodule.append('')
                    if args.dl4:
                        nodule.append(str(preds_malignancy[i][0]))
                        if dataset == 'LUNA':
                            nodule.append(str(y_malignancy[i]))
                        else:
                            nodule.append('')
                    if args.dl6:
                        nodule.append(str(preds_lobulation[i][0]))
                        if dataset == 'LUNA':
                            nodule.append(str(y_lobulation[i]))
                        else:
                            nodule.append('')
                    if args.dl8:
                        nodule.append(str(preds_spiculation[i][0]))
                        if dataset == 'LUNA':
                            nodule.append(str(y_spiculation[i]))
                        else:
                            nodule.append('')
                    f.write(','.join(nodule)+'\n')
                    #f.write('%s,%d,%d,%d,%.3f,%.5f,%d\n' % (patientName, nslice, r.centroid[0], r.centroid[1], r.equivalent_diameter,preds[i],y[i]))

        except Exception as e:
            logging.error("Error processing result, skipping. %s" % str(e))

    if args.roi_statistics_csv:
        stats_roi_pd.to_csv(args.roi_statistics_csv)

    f.close()
