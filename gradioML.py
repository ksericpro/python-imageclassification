import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import gradio
import gradio as gr

DATADIR = os.path.abspath(os.getcwd()) + '/data'
path = os.path.join(DATADIR, 'pets')    #path to cats or dogs dir
first_img_path = os.listdir(path)[0]
img_array = cv2.imread(os.path.join(path, first_img_path), cv2.IMREAD_GRAYSCALE)
plt.imshow(img_array, cmap = "gray")
plt.show()
#show image shape
print('The image shape is {}'.format(img_array.shape))