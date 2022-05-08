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
from sklearn import model_selection
import tensorflow as tf
from keras import layers
from keras.models import Sequential
# import tkinter as tk
# from tkinter import filedialog
# Added 18.11.2021
# import random
# from keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.utils import to_categorical

# Config


# Functions
def adjustImg(img):
    mean = np.mean(img)
    for row in img:
        for pixel in row:
            if pixel > mean:
                pixel = 255
    return img


def convert_imgs2arr(dir_name, dicom=False) -> list:
    arr: list = []
    dir = glob.glob(dir_name + '/*')
    for filename in dir:
        im = np.array(cv2.resize(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY), (96, 96)))
        # im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_TOZERO, 11, 2)
        im = adjustImg(im)
        arr.append(im)
    return arr


# Main script
normal_list = convert_imgs2arr('NORMAL')
pneumo_list = convert_imgs2arr('PNG', True)
all_imgs = np.concatenate((normal_list, pneumo_list), axis=0)
dataset = np.arange(0, len(all_imgs))
normal_labels = [0 for i in range(0, len(normal_list))]
pneumo_labels = [1 for i in range(0, len(pneumo_list))]
all_labels = np.concatenate((normal_labels, pneumo_labels), axis=0)

labels = np.arange(0, len(all_imgs))
for i in range(0, len(dataset)):
    dataset[i] = i
ind_train, ind_test = model_selection.train_test_split(dataset, test_size=.3, random_state=0)
print(ind_train)
def getImgsArrs(source):
    labels = []
    vals = []
    for i in source:
        labels.append(all_labels[i])
        vals.append(all_imgs[i])
    return vals, labels

## Train ##
imgs_train, labels_train = getImgsArrs(ind_train)
## Test ##
imgs_test, labels_test = getImgsArrs(ind_test)
# network
nOfClasses = 2
model = Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(96,96,1)),
        layers.Conv2D(16,3,padding='same', activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(32,3,padding='same', activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(64,3,padding='same', activation='relu'),
        layers.MaxPool2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(64, activation='sigmoid'),
        layers.Dense(nOfClasses, activation='sigmoid')
])
nOfEpochs = 100

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])#tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
#training our model
# our_model = model.fit(imgs_train, labels_train, epochs = nOfEpochs,validation_split=0.1,verbose=1)
