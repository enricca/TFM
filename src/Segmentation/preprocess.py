"""
File for preprocessing the datasets. It gets as input the DCOMS path, and converts it to numpy.
Datasets accepted: ['isbi', 'dsb', 'luna']
We preprocess the masks of the nodules.
Example usage:
python lung_cancer_isbi18/src/preprocess.py --input_folder ~/../../media/shared/datasets/ISBI/ISBI-deid-TRAIN-DICOM --output_folder ~/../../media/shared/datasets/ISBI/preprocessed --nodule_info ~/../../media/shared/datasets/ISBI/ISBI-deid-TRAIN-SegMASK/ --pipeline isbi
"""
#python lung_cancer_isbi18/src/preprocess.py --input_folder /home/enric/Desktop/TFM/DATA_LUNA/all_subsets --output_folder /home/enric/Desktop/TFM/DATA_LUNA/preprocessed --nodule_info /home/enric/Desktop/TFM/DATA_LUNA/CSVFILES/annotations.csv --pipeline luna


import argparse
import logging
import os
import sys
from glob import glob
from time import time
import nibabel

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import scipy.sparse
from joblib import Parallel, delayed
from utils import lung_segmentation, plotting, preprocessing, reading, readingXMLSegmentation
import traceback
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M:%S')

# Define parametres
COMMON_SPACING = [2, 0.7, 0.7]

#HARDCODED
# pIdToXML = pd.read_csv('/home/gabriel/xmlParsing/pIdToXML.csv')
pIdToXML = pd.read_csv('/home/enric/Desktop/TFM/DATA_LUNA/CSVFILES/patientIdToCSV2.csv')

def process_filename(patient_file,
                     output_folder,
                     pipeline='dsb',
                     nodule_info=None,
                     resample = True,
                     extra_features = True,
                     main_slice = False,
                     resegment_nodules=None,
                     sphere_nodules = False,
                     ignore_non_nodules = False
                     ):
    malign = extra_features
    lobulation = extra_features
    spiculation = extra_features
    # truenoduleslicepath = '/home/shared/isbi_groundtruth_nodules_trueNoduleSlice.csv'
    #Just needed if DSB or ISBI
    # truenoduleslicepath = '/media/shared/datasets/ISBI/isbi_groundtruth_nodules_trueNoduleSlice.csv'
    
    """
    Preprocess a folder containing a patient scan (several slices) of a patient.

    1) Resample to homogenous resolution (if needed)
    2) Segment
    3) Convert to numpy, and save the whole scan in output folder.
    
    IF SPACING DF IS NOT NONE IT WILL NOT RESAMPLE
    """
    nodule_mask = None
    malign_mask = None
    lob_mask = None
    spic_mask = None
    main_nodule_mask = None
    logging.info('Processing patient: %s' % patient_file)

    # GENERATE PATIENT ID
    try:
        patient_file = os.path.normpath(patient_file) 
        _, splitName1 = os.path.split(patient_file)                    # Patient folder (ISBI, DSB) or file (LUNA)
        _, splitName2 = os.path.split(os.path.split(patient_file)[0])  # Patient parent folder (for ISBI)
        if pipeline == 'dsb':
            pat_id = splitName1
        elif pipeline == 'isbi':
            pat_id = 'PID-' + splitName2 + '-' + splitName1
        elif pipeline == 'luna':
            pat_id = splitName1.split('.')[-2]
        else:
            raise Exception('Pipeline %s not recognized!' % pipeline)
    except Exception as e:  # Some patients have no data, ignore them
        logging.error('Exception', e)
        return
    #print pat_id
    outputFile = os.path.join(output_folder, "%s_%s.npz") % (pipeline, pat_id)   

    # SKIP PATIENT IF ALREADY COMPUTED
    if os.path.isfile(outputFile):
        logging.info('Patient %s already processing. Ignoring' % patient_file)
        return
    
    # READ CT SCANS
    try:
        if pipeline == 'dsb' or pipeline == 'isbi':
            patient = reading.load_scan(patient_file)
            patient_pixels = preprocessing.get_pixels_hu(
                patient)  # From pixels to HU
            originalSpacing = reading.dicom_get_spacing(patient)
            if pipeline == 'isbi': # Read nodule mask
                try:
                    segmentationFileName = splitName2 + '_Y' + splitName1 + '.dcm'  
                    nodule_mask = reading.create_mask_from_dso(
                        os.path.join(nodule_info, segmentationFileName), patient) 
                    if main_slice:
                        main_nodule_mask = reading.main_nodule_mask(nodule_mask, truenoduleslicepath, 'isbi_' + pat_id)
                except:
                    nodule_mask = nib.load(os.path.join(nodule_info,splitName2 + '_Y' + splitName1 + '.nii.gz'))
                    #
                    if img.affine[2,2] < 0:
                        print 'invert'
                    nodule_mask

        elif pipeline == 'luna':
            patient = sitk.ReadImage(patient_file)
            patient_pixels = sitk.GetArrayFromImage(
                patient)  # indexes are z,y,x
            originalSpacing = [
                patient.GetSpacing()[2],
                patient.GetSpacing()[0],
                patient.GetSpacing()[1]
            ]
            pat_id = patient_file.split('.')[-2]
            # load nodules
            seriesuid = splitName1.replace('.mhd', '')
            #print( pIdToXML['pId'] == seriesuid )
            xmlPath = pIdToXML[pIdToXML['pId'] == seriesuid].iloc[0].xmlPath
            # quan dóna error és perque no hi ha match entre la llista de pacients i els fitxers
            # que hi ha al input_folder
            print('xmlPath : ', xmlPath)
            nodules = nodule_info[nodule_info["seriesuid"] == seriesuid]  # filter nodules for patient
            try:
                nodule_mask = readingXMLSegmentation.create_mask_from_xml(img=patient, nodules=nodules, xmlPath = xmlPath)
            except Exception as e:
                logging.error(
                    'PATIENT DOES NOT EXIST {}')
                traceback.print_exc()
                logging.error('Exception {}'.format(e))
                return

            if nodule_mask is None:  # create virtual nodule mask for coherence
                nodule_mask = np.zeros(patient_pixels.shape, dtype=np.int)
                if ignore_non_nodules:
                    logging.info('Patient ignored')
                    return
            if malign:
                malign_mask = readingXMLSegmentation.create_mask_from_xml(img = patient, nodules = nodules , channel = 'malignancy',xmlPath = xmlPath)
                if malign_mask is None:
                    print 'No malignity'
                    malign_mask = np.zeros(patient_pixels.shape, dtype=np.int)
            if lobulation:
                lob_mask = readingXMLSegmentation.create_mask_from_xml(img = patient, nodules = nodules, channel = 'lobulation',xmlPath = xmlPath)
                if lob_mask is None:
                    print 'No lobulation'
                    lob_mask = np.zeros(patient_pixels.shape,dtype=np.int)
            if spiculation:
                spic_mask = readingXMLSegmentation.create_mask_from_xml(img = patient, nodules = nodules,  channel = 'spiculation' ,xmlPath = xmlPath)
                if spic_mask is None:
                    print 'No spiculation'
                    spic_mask = np.zeros(patient_pixels.shape,dtype=np.int)
        else:
            raise Exception('Pipeline %s not recognized!' % pipeline)
    except Exception as e:  # Some patients have no data, ignore them
        logging.error(
            'There was some problem reading patient {}. Ignoring and live goes on.'.
            format(patient_file))
        traceback.print_exc()
        logging.error('Exception {}'.format(e))
        return
    
    # SET BACKGROUND: set to air parts that fell outside
    patient_pixels[patient_pixels < -1500] = -2000
    
    if resample:

        # RESAMPLING
        pix_resampled, new_spacing = preprocessing.resample(
            patient_pixels, spacing=originalSpacing, new_spacing=COMMON_SPACING)
        
        
        if nodule_mask is not None:
            nodule_mask, new_spacing = preprocessing.resample(
                nodule_mask, spacing=originalSpacing, new_spacing=COMMON_SPACING)
        logging.info('Resampled image size: {}'.format(pix_resampled.shape))
        if main_slice:
            if main_nodule_mask is not None:
                main_nodule_mask, new_spacing = preprocessing.resample(main_nodule_mask, spacing=originalSpacing, new_spacing=COMMON_SPACING)
        
        if malign:
            if malign_mask is not None:
                print('malign')
                malign_mask, new_spacing = preprocessing.resample(malign_mask, spacing=originalSpacing, new_spacing=COMMON_SPACING)
        if lobulation:
            if lob_mask is not None:
                lob_mask, new_spacing = preprocessing.resample(lob_mask, spacing = originalSpacing, new_spacing=COMMON_SPACING)
        if spiculation:
            if spic_mask is not None:
                spic_mask, new_spacing = preprocessing.resample(spic_mask, spacing = originalSpacing, new_spacing=COMMON_SPACING)    
                
        # LUNG SEGMENTATION (if TH1 fails, choose TH2)
        #lung_mask = lung_segmentation.segment_lungs(image=pix_resampled, fill_lung=True, method="watershed")
        lung_mask = lung_segmentation.segment_lungs(
            image=pix_resampled, fill_lung=True, method="thresholding1")
        if not lung_segmentation.is_lung_segmentation_correct(lung_mask):
            logging.info(
                "Segmentation TH1 failed for %s. Trying method 2" % patient_file)
            lung_mask = lung_segmentation.segment_lungs(
                image=pix_resampled, fill_lung=True, method="thresholding2")

        # CROPPING to 512x512
        pix = preprocessing.resize_image(
            pix_resampled, size=512)  # if zero_centered: -0.25
        lung_mask = preprocessing.resize_image(lung_mask, size=512)
        if nodule_mask is not None:
            logging.info('Nodule mask with %f voxels' % np.sum(nodule_mask))
            nodule_mask = preprocessing.resize_image(nodule_mask, size=512)
        else:
            logging.info('No nodule mask')
            
        if main_slice:
            if main_nodule_mask is not None:
                logging.info('Main nodule mask done')
                main_nodule_mask = preprocessing.resize_image(main_nodule_mask,size=512)
        if malign:
            if malign_mask is not None:
                logging.info('Malignancy mask done' )
                malign_mask = preprocessing.resize_image(malign_mask,size=512)
            else:
                logging.info('No malignancy mask')
        if lobulation:
            if lob_mask is not None:
                logging.info('Lobulation mask done')
                lob_mask = preprocessing.resize_image(lob_mask,size=512)
            else:
                logging.info('No lobulation mask')
        if spiculation:
            if spic_mask is not None:
                logging.info('Spiculation mask done')
                spic_mask = preprocessing.resize_image(spic_mask,size=512)
            
        logging.info('Cropped image size: {}'.format(pix.shape))
    #Mantain cpacing, store it in a df
    else:
        pix = patient_pixels
        lung_mask = lung_segmentation.segment_lungs(
            image=pix, fill_lung=True, method="thresholding1")
        
        if not lung_segmentation.is_lung_segmentation_correct(lung_mask):
            logging.info(
                "Segmentation TH1 failed for %s. Trying method 2" % patient_file)
            lung_mask = lung_segmentation.segment_lungs(
                image=pix, fill_lung=True, method="thresholding2")
#     logging.info('Segmentation finished, volume: {} liters'.format(np.sum(lung_mask) * np.prod(COMMON_SPACING) / 1e6))

    if nodule_mask is not None and resegment_nodules:
        nodule_mask = reading.resegment_nodules(pix, nodule_mask, resegment_nodules)

    # STACK and save results
    if main_slice:
        output = np.stack((pix, lung_mask,
                           nodule_mask,main_nodule_mask)) if nodule_mask is not None else np.stack(
                               (pix, lung_mask))
        np.savez_compressed(outputFile, output)
#         np.savez_compressed(outputFile[:-4]+'_mainnodulemask.npz',main_nodule_mask)
    elif not extra_features:
        output = np.stack((pix, lung_mask,
                           nodule_mask)) if nodule_mask is not None else np.stack(
                               (pix, lung_mask))
        np.savez_compressed(outputFile, output)
    else:
        output = np.stack((pix,lung_mask, 
                           nodule_mask, malign_mask,lob_mask,spic_mask)) if nodule_mask is not None else np.stack(
                               (pix, lung_mask))
        np.savez_compressed(outputFile, output)

    logging.info('Patient finished')
    return {'pId' : pat_id, 's1' : originalSpacing[0], 's2' : originalSpacing[1], 's3' : originalSpacing[2]}

def preprocess_files(file_list,
                     output_folder,
                     pipeline='dsb',
                     nodule_info=None,
                     parallel=False,
                     resample = True,
                     df_spacing = None,
                     resegment_nodules=None,
                     sphere_nodules = False,
                     ignore_non_nodules = False,
                     extra_features = False,
                    main_slice = False):
    tstart = time()
    if parallel:
        n_jobs = 3
        Parallel(n_jobs=n_jobs)(delayed(process_filename)(
            filename, output_folder, pipeline, nodule_info, resample, df_spacing, sphere_nodules = sphere_nodules, ignore_non_nodules = ignore_non_nodules)
                                for filename in file_list)
    else:
        for filename in file_list:
            sp = process_filename(filename, output_folder, pipeline, nodule_info, resample,
                                  resegment_nodules=resegment_nodules, sphere_nodules = sphere_nodules,
                                  ignore_non_nodules = ignore_non_nodules,extra_features = extra_features,
                             main_slice = main_slice)
            if df_spacing is not None:
                print sp
                df_spacing = df_spacing.append(sp, ignore_index = True)
    logging.info("Finished preprocessing in %.3f s" % (time() - tstart))
    return df_spacing

if __name__ == '__main__':
    # python preprocess.py --input_folder=/mnt/hd2/stage2/ --output_folder=/mnt/hd2/preprocessed_stage2 --pipeline=dsb
    parser = argparse.ArgumentParser(
        description='Preprocess patient files in parallel')
    parser.add_argument('--input_folder', help='input folder')
    parser.add_argument('--output_folder', help='output folder')
    parser.add_argument('--resegment_nodules', default='', help='output folder')
    parser.add_argument('--sphere', action = 'store_true', help='segment nodules as spheres')
    parser.add_argument('--ignore_no_nodules', action = 'store_true', help='ignores patients with no marked nodules')
    parser.add_argument('--extra_features', action = 'store_true', help='load malignancy, spiculation, etc')
    parser.add_argument('--main_slice', action = 'store_true', help='load only the main slice')

    parser.add_argument(
        '--pipeline',
        default='dsb',
        help='pipeline to be used (dsb, luna or isbi)')
    parser.add_argument(
        '--nodule_info',
        help=
        'in case of luna pipeline, csv with nodules annotations. In case of ISBI, path of the folder with segmentations'
    )
    parser.add_argument('--parallel',action = 'store_true', help='Parallel computing (WARNING, HIGH USAGE OF MEMORY)')
    parser.add_argument('--no_resampling',action = 'store_true', help='Toogle resampling option.')
    parser.add_argument('--original_spacing_csv', default = '', help='Toogle resampling option.')

    args = parser.parse_args()

    # CHECK INPUT FOLDER
    if not os.path.isdir(args.input_folder): 
        logging.error('Input folder does not exist.')
        sys.exit()

    print(__name__)
    print(args)
    print(args.pipeline)
    # CREATE LIST OF PATIENT FILES
    patient_files = []
    nodule_info = None
    if args.pipeline == 'dsb':
        patient_files = [
            os.path.join(args.input_folder, p)
            for p in os.listdir(args.input_folder)
        ]
    elif args.pipeline == 'luna':
        patient_files = glob(
            args.input_folder + '/*.mhd')  # patients from subset
        nodule_info = pd.read_csv(args.nodule_info)

    elif args.pipeline == 'isbi':
        patient_files = os.listdir(args.input_folder)
        patient_files = filter(
            lambda s: s.isdigit(),
            patient_files)  #The patients ids have to be a digit
        nodule_info = args.nodule_info
        #Two scans per patient, load them
        patient_files = [[
            os.path.join(args.input_folder, pId, p)
            for p in os.listdir(os.path.join(args.input_folder, pId))
        ] for pId in patient_files]
        patient_files = reduce(lambda s1, s2: s1 + s2, patient_files)

    # PREPROCESS PATIENTS
    resampling = not args.no_resampling
    df_spacing = pd.DataFrame() if args.original_spacing_csv else None
    if args.parallel and df_spacing is not None:
        logging.info("Deactivating parallel")
        args.parallel = False

    df_spacing = preprocess_files(
        file_list=patient_files,
        output_folder=args.output_folder,
        pipeline=args.pipeline,
        nodule_info=nodule_info,
        parallel = args.parallel,
        resample = resampling,
        df_spacing = df_spacing,
        resegment_nodules=args.resegment_nodules,
        sphere_nodules = args.sphere,
        ignore_non_nodules = args.ignore_no_nodules,
        extra_features = args.extra_features,
        main_slice = args.main_slice
    )
    if args.original_spacing_csv:
        df_spacing.to_csv(args.original_spacing_csv)

