import numpy as np
import scipy.ndimage

MIN_BOUND = -1000.0  # Normalization
MAX_BOUND = 400.0  # Normalization
PIXEL_MEAN = 0.25  # centering


def get_pixels_hu(slices):
    """
    Given an array of slices from the DICOM, returns and array of images, 
    converting pixel values to HU units.
    """
    # slices = __set_canonical_position__(slices)

    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


def __set_canonical_position__(slices):
    flag_for_sn = False
    print(slices[0])
    # Yet another sanity check
    try:
        try:
            print('PatientOrientation: {}'.format(slices[0].PatientOrientation))
        except:
            print('DICOM PatientOrientation field not found')

        try:
            print('PositionReferenceIndicator: {}'.format(slices[0].PositionReferenceIndicator))
            if slices[0].PositionReferenceIndicator != 'SN':
                """
                Gabriel: For more information about this field, see
                http://dicom.nema.org/medical/Dicom/2015a/output/chtml/part03/sect_C.7.6.2.html#sect_C.7.6.2.1.1

                I really really love working with DICOMS <3<3<3
                """
                print('The position reference is not SN but {}'.format(slices[0].PositionReferenceIndicator))
            else:
                flag_for_sn = True
        except:
            print('DICOM PositionReferenceIndicator field not found')

        try:
            print('ImageOrientationPatient: {}'.format(slices[0].ImageOrientationPatient))
        except:
            print('DICOM ImageOrientationPatient field not found')

        try:
            # Reversing stack in z direction
            delta = slices[0].SliceLocation - slices[-1].SliceLocation
            print('slices[0].SliceLocation-slices[-1].SliceLocation = {}'.format(delta))

            if flag_for_sn and delta < 0:
                print('Reorienting dataset...')
                slices = slices[::-1]
                for x in range(len(slices)):
                    slices[x].pixel_array[:, :] = slices[x].pixel_array[:, ::-1]
        except:
           print('DICOM SliceLocation field not found')

    except:
        print('This is not a CT image stack')

    return slices


def resample(image, spacing, new_spacing=[1, 1, 1], method='nearest'):
    """
    Resample image given spacing (in mm) to new_spacing (in mm).
    """
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    if method == 'cubic':
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, order=3)
    elif method == 'quadratic':
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, order=2)
    elif method == 'linear':
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, order=1)
    elif method == 'nearest':
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, order=0)
    else:
        raise NotImplementedError("Interpolation method not implemented.")

    return image, new_spacing


def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def zero_center(image):
    image -= PIXEL_MEAN
    return image


def extend_image(img, val=-0.25, size=512):
    """
    Resize <img> centered to <size> filling with extra values <val>
    """
    result = np.zeros((img.shape[0], size, size))
    result.fill(val)

    x = (size - img.shape[1])/2
    y = (size - img.shape[2])/2
    result[:, x:x+img.shape[1], y:y+img.shape[2]] = img
    return result


def crop_image(img, size=512):
    cx = img.shape[1]/2
    cy = img.shape[2]/2
    new_img = img[:, cx - size/2:cx + size/2, cy - size/2:cy + size/2]
    return new_img


def resize_image(img, size=512):
    x = img.shape[1]
    if x < size:
        return extend_image(img, val=img[0,0,0], size=size)
    elif x > size:
        return crop_image(img, size=size)
    else:
        return img
