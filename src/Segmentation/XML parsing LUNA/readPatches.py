# reading XMLs from LUNA and extracting the masks of the nodules

import sys
sys.path.append('Code/')

import SimpleITK as sitk
import readingXMLSegmentation
import importlib
reload(readingXMLSegmentation)
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import skimage, skimage.measure
import radiomics
import numpy

fe = radiomics.featureextractor.RadiomicsFeaturesExtractor()

csvPath = 'DATA_LUNA/CSVFILES/annotations.csv'
nodules = pd.read_csv(csvPath)
xmlPath = 'LIDC_xml/LIDC-XML_radiologist_anotations/tcia-lidc-xml/'
imagePath = 'DATA_LUNA/all_subsets/'

patients = set(nodules.seriesuid.values)

#Get a scan, get its segmentations, and link it to malignancy.
#Extract a numpy array with (image), and a csv with malignancy, spicatulation, etc, radiomics property, and identifier.

# link patient ID to xml corresponding to the mask of the nodule
readingXMLSegmentation.generate_csv_xml_path(xmlPath, patients, 'patientIdToCSV2.csv')

patientsToCSV = pd.read_csv('patientIdToCSV2.csv')
patientsToCSV.index = patientsToCSV.pId

# this is just an example: it takes just the first patientID
pIdTest = patientsToCSV.index.values[0]
xmlPath = patientsToCSV.loc[pIdTest , 'xmlPath']

#Reading
image = sitk.ReadImage(os.path.join(imagePath, pIdTest +'.mhd'))
nodulesPatient = nodules.loc[nodules['seriesuid'] == pIdTest]

reload(readingXMLSegmentation)

# get the Nodule Mask from XML
mask = readingXMLSegmentation.getNoduleMaskFromXML(xmlPath, image)
maskSitk = sitk.GetImageFromArray(mask)
maskSitk.SetSpacing(image.GetSpacing())
maskSitk.SetOrigin(image.GetOrigin())

#Link with nodules list
labels = skimage.measure.label(mask)
rProps = skimage.measure.regionprops(labels)

def centroidDifferences(nodule, centroid):
    return np.linalg.norm([nodule.coordX - centroid[0],nodule.coordY - centroid[1],nodule.coordZ - centroid[2]])


def extract_radiomic_features(patient_id, img, nodule_mask, spacing = [1,1,1], add ={}):

    settings = {}
    settings['binWidth'] = 25
    settings['resampledPixelSpacing'] = None
    # settings['resampledPixelSpacing'] = [3, 3, 3]  # This is an example for defining resampling (voxels with size 3x3x3mm)
    settings['interpolator'] = 'sitkBSpline'
    settings['verbose'] = True

    #from radiomics import featureextractor
    extractor = radiomics.featureextractor.RadiomicsFeaturesExtractor(**settings)

    labels, nLabels =skimage.measure.label(nodule_mask, return_num = True)
    allNodules = []
    print('nlabels = ', nLabels)

    for i in xrange(1,nLabels + 1):

        # change npz data to integer (for pyradiomics)
        mask = (labels == i).astype(np.int8)
        print ('label %d, number of voxels = %d' % (i, np.sum(mask)))

        # change image format to SimpleITK (for pyradiomcis)
        imgSitk = sitk.GetImageFromArray(img)
        imgSitk.SetSpacing(spacing)
        maskSitk = sitk.GetImageFromArray(mask)
        maskSitk.SetSpacing(spacing)
        # exctract features
        print('Calculating features')
        featureVector = extractor.execute(imgSitk, maskSitk)

        # create dataframe
        feature_list = featureVector.items()[12:]
        patient_radiomics = pd.DataFrame(feature_list)

        patient_radiomics = patient_radiomics.transpose()
        patient_radiomics.columns = patient_radiomics.iloc[0]
        patient_radiomics = patient_radiomics.reindex(patient_radiomics.index.drop(0))

        patient_radiomics['patientid'] = patient_id
        patient_radiomics['num_features'] = len(feature_list)
        patient_radiomics['nodule_id'] = i
        for k, v in add.iteritems():
            patient_radiomics[k] = np.max(v[np.where(mask)])
        allNodules.append(patient_radiomics)
    return allNodules


if len(rProps) != len(nodulesPatient):
    errors = []
    errors.append(pIdTest)
    #print('Error in patient', pId, 'number of anotated nodules %d / in csv %d ' % (len(rProps), len(nodulesPatient)) )
    print('Error in patient', pIdTest, 'number of anotated nodules %d / in csv %d ' % (len(rProps), len(nodulesPatient)))
for r in rProps:
    centroidPatientCoordinatesrProps = r.centroid[::-1] * np.array(image.GetSpacing()) + np.array(image.GetOrigin())
    for i,nod in nodulesPatient.iterrows():
        print(centroidDifferences(nod, centroidPatientCoordinatesrProps) )

imageNP = sitk.GetArrayViewFromImage(image)
sys.path.append('Code/pyradiomics/radiomics/')
mask2 = sitk.GetImageFromArray(mask)
# extract radiomics features
fe = radiomics.featureextractor.RadiomicsFeaturesExtractor()
fe.computeFeatures(image, mask2, 'sitk')

plt.figure(figsize = (20, 20))
plt.imshow(imageNP[123, :, :], cmap = 'gray')
#plt.scatter(130, 190, c = 'r')
plt.contour(mask[123, :, :])


