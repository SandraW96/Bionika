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
        im = np.array(cv2.resize(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY), (96, 96)))
        arr.append(im)
    return arr


def adjustImgList(imglist) -> list:
    for i in imglist:
        for row in i:
            for column in row:
                column*=2
                if column < 75:
                    column = 0
    return imglist


# Main script
normal_list = convert_imgs2arr('NORMAL')
pneumo_list = convert_imgs2arr('PNG', True)

pneumo_list = adjustImgList(pneumo_list)
normal_list = adjustImgList(normal_list)