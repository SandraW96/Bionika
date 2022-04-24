# Import libs
import cv2
import numpy as np
import glob2 as glob
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import os
import epydoc
import matplotlib as plt
from matplotlib import pyplot


# Config


# Functions
def convert_imgs2arr(dir_name, dicom=False) -> list:
    arr: list = []
    dir = glob.glob(dir_name + '/*')
    for filename in dir:
        if dicom:
            im = pydicom.read_file(filename, force=True)
            # if voi_lut:
            #     data = apply_voi_lut(dicom.pixel_array, dicom)
            #sthelse = im.pixel_array
        else:
            im = np.array(cv2.imread(filename))
        arr.append(im)
    return arr


# for f in test_list[:10]:   # remove "[:10]" to convert all images
#     ds = pydicom.read_file(inputdir + f) # read dicom image
#     img = ds.pixel_array # get image array

# Main script
# normal = convert_imgs2arr('NORMAL')
pneumothorax = convert_imgs2arr('PNEUMOTHORAX', True)
