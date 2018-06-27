"""
File for reading the manual segmentation done by clinicians
Functions used in the file: readPatches.py
"""

import xml.etree.ElementTree as ET
import numpy as np
import skimage, skimage.draw
import os
import matplotlib.pyplot as plt
import pandas
import collections
import SimpleITK as sitk

def generate_csv_xml_path(xmlPath, patients,  outputCsvPath):
    """
    Generates a path that links series instanceUID with the xml file
    """
    paths = {}
    for subfolder in filter(lambda s: not s.startswith('.') and os.path.isdir(os.path.join(xmlPath, s)), os.listdir(xmlPath)):
        path = os.path.join(xmlPath, subfolder)
        for p in filter(lambda s: not s.startswith('.') and s.endswith('.xml'), os.listdir(path)):
            tree = ET.parse(os.path.join(path, p))
            root = tree.getroot()
            nDomain = root.getchildren()[0].tag.split('}')[0] + '}'
            try:
                patientId = str(root.find('%sResponseHeader/%sSeriesInstanceUid' % (nDomain, nDomain)).text)
            except:
                #Funny clinicians with 
                patientId = str(root.find('%sResponseHeader/%sCTSeriesInstanceUid' % (nDomain, nDomain)).text)
            if not  patientId in patients:
                continue
            #patientIdNumber = patients.loc[patients['Series UID'] == patientId , 'PatId'].values[0]
            paths[patientId] = os.path.join(path, p)
    pd = pandas.DataFrame.from_dict({'pId' : paths.keys(), 'xmlPath' : paths.values()})
    pd.to_csv(outputCsvPath)
    
def ball(rad, center, spatialScaling=[1, 1, 1]):
    """
    Creates a ball of radius R, centered in the coordinates center
    @param rad: radius of the ball, in mm
    @param center: center of the ball (slice, x, y) in pixel coordinates

    @spatialSpacing

    returns a list of coordinates (x, y, z) that are in the ball of radius r centered in 0.
    """

    # Generate the mesh of candidates
    r = np.ceil(rad / spatialScaling).astype(int)  # anisotropic spacing
    x, y, z = np.meshgrid(
        xrange(-r[0], r[0] + 1),
        xrange(-r[1], r[1] + 1), xrange(-r[2], r[2] + 1))
    mask = (x * spatialScaling[0])**2 + (y * spatialScaling[1])**2 + (
        z * spatialScaling[2])**2 <= rad**2
    return np.stack((z[mask] + center[2], x[mask] + center[0],
                     y[mask] + center[1])).T
def label_from_point(start, mask, directions = np.array([[1,0, 0], [0,1,0], [0, 0, 1], [-1,0,0],[0, -1, 0], [0,0, -1]])):
    """
    gets all the points in the same  connected component that contains start. 
    """
    points = []
    toExplore = [start]
    explored = np.zeros(mask.shape, dtype = bool)
    while toExplore:
        p = toExplore.pop()
        if explored[tuple(p)] or not mask[tuple(p)]:
            continue
        points.append(p)
        explored[tuple(p)] = True
        for d in directions:
            toExplore.append(p + d)
    return points

def noduleToMask(roi, shape, nDomain = ''):
    #Parse z
    z = float(roi.find('%simageZposition'% nDomain).text)
    inclusion = roi.find('%sinclusion' % nDomain).text == 'TRUE'
    
    #Parse x, y
    noduleROIX =  roi.findall('%sedgeMap/%sxCoord' % (nDomain, nDomain))
    noduleROIY =  roi.findall('%sedgeMap/%syCoord'% (nDomain, nDomain))
    x, y = np.array([float(x.text) for x in noduleROIX]), np.array([float(y.text) for y in noduleROIY])
    rr, cc = skimage.draw.polygon(x, y, shape )
    mask = np.zeros(shape, dtype = np.int16)
    mask[cc, rr] = 1
    if not inclusion:
        mask *= -1
    return mask, z

def parseNodule(nodule, allRois, shape, nDomain = ''):
    for r in nodule.findall('%sroi' %nDomain):
        mask, z = noduleToMask(r, shape, nDomain)
        #Check if ROI is include or exclude
        allRois[z].append(mask)
        
        
def getNoduleMaskFromXML(xmlPath, sitkImage):
    """
    Creates the nodule mask from the xml file and the sitkImage
    """
    size = sitkImage.GetSize()
    
    mask = np.zeros([size[2], size[0], size[1]], dtype = np.int16)
    tree = ET.parse(xmlPath)
    root = tree.getroot()
    nDomain = root.getchildren()[0].tag.split('}')[0] + '}'
    nodules = root.findall('%sreadingSession/%sunblindedReadNodule' % (nDomain, nDomain))
    slices = collections.defaultdict(list)
    
    for n in nodules:
        parseNodule(n, slices, size[:2], nDomain = nDomain)
        
    for f in slices.keys():
        nSlice = int(np.round((f - sitkImage.GetOrigin()[2])/ sitkImage.GetSpacing()[2]))
        mask[nSlice, :, :] = np.sum(slices[f], axis = (0)) >= 3
    return mask

def create_mask_from_xml(img, xmlPath, nodules, channel = 'nodule'):
    """
    Creates a mask from nodule csv. If 
    """
    if len(nodules) == 0:
        return None

    height, width, num_z = img.GetSize()
    noduleMask = getNoduleMaskFromXML(xmlPath, img)
    mask = np.zeros(noduleMask.shape, dtype =  np.int16)
    origin = np.array(
        img.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
    spacing = np.array(
        img.GetSpacing())  # spacing of voxels in world coor. (mm)

    # row = nodules.iloc[0]
    for index, row in nodules.iterrows():
        node_x = row["coordX"]
        node_y = row["coordY"]
        node_z = row["coordZ"]
        diam = row["diameter_mm"]
        center = np.array([node_x, node_y, node_z])  # nodule center (in mm)

        center = np.rint((center-origin)/spacing) #nodule center (in pixels)
        center = np.array([center[2], center[1], center[0]], dtype = np.int16)
        if diam > 3:
            points = label_from_point(center, noduleMask)
            value = 1. if channel == 'nodule' else np.int16(np.round(row[channel]*84))
            points = np.array(points)
            if len(points):
                mask[points[:,0], points[:, 1], points[:,2]] = value
        #Ignore small points 
        #else:
        #    print 'small'
        #    points = ball(diam/2, center, spacing)

    return mask

