import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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
import random

CATEGORIES = ['dogs', 'cats']
IMG_SIZE = 100 

DATADIR = os.path.abspath(os.getcwd()) + '/PetImages'
path = os.path.join(DATADIR, category)    #path to cats or dogs dir
first_img_path = os.listdir(path)[0]
img_array = cv2.imread(os.path.join(path, first_img_path), cv2.IMREAD_GRAYSCALE)
plt.imshow(img_array, cmap = "gray")
plt.show()
#show image shape
print('The image shape is {}'.format(img_array.shape))

#create create array of data
data = []
def create_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  
        class_num = CATEGORIES.index(category)    
        for img in os.listdir(path):
            try: 
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([new_array, class_num])
            except Exception as e:
                pass
create_data()
#randomly shuffle the images
random.shuffle(data)
#separate features and labels
X = []
y = []
for features, label in data:
    X.append(features)
    y.append(label)
#neural network takes in a numpy array as the features and labels so convert from list to array and change shape
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

#show first feature image X
first_feature = X[0]
plt.imshow(first_feature, cmap = 'gray')
print('The image shape is {}'.format(first_feature.shape))

#normalize images
X = X/255.0

#separate training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('the shape of training features is {}'.format(X_train.shape))
print('the shape of training labels is {}'.format(y_train.shape))
print('the shape of test features is {}'.format(X_test.shape))
print('the shape of test labels is {}'.format(y_test.shape))

#create model
model = Sequential()
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.1))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, input_shape = X.shape[1:], activation = 'relu'))
#output layer
model.add(Dense(2, activation = 'softmax'))

#compile the model
model.compile(loss="sparse_categorical_crossentropy",
             optimizer="adam",
             metrics=['accuracy'])

#fit model
history = model.fit(X_train, y_train, epochs=5, validation_split=0.1)

#show learning curves
#mean training loss and accuracy measured over each epoch
#mean validation loss and accuracy measured at the end of each epoch
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1) # set the vertical range to [0-1]
plt.show()

#use predict_classes() to find the class with the highest probability
y_pred = model.predict_classes(X_test)
print("Performance Summary of Sequential Neural Network on test data:")
#show classification report
print(metrics.classification_report(y_test, y_pred))
#show confusion matrix
print(metrics.confusion_matrix(y_test, y_pred))

#show first 5 correctly identified test images with predicted labels and probabilities
fig, ax = plt.subplots(1,5,figsize=(20,20))
class_names = ["Dog", "Cat"]
for i, correct_idx in enumerate(correct_indices[:5]):
    ax[i].imshow(X_test[correct_idx].reshape(100,100),cmap='gray')
    ax[i].set_title("{} with probabililty of {}%".format(class_names[y_pred[correct_idx]], int(max(y_proba[correct_idx]