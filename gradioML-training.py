import os
import random

# For Data Exploration, ETL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# For Gradio
import gradio
import gradio as gr

# For Data Visualization
import cv2
import seaborn as sns

# For Model Building
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential, Model # Sequential API for sequential model
from tensorflow.keras.layers import Dense, Dropout, Flatten # Importing different layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Input, LeakyReLU, Activation
from tensorflow.keras import backend
from tensorflow.keras.utils import to_categorical # To perform one-hot encoding
from tensorflow.keras.optimizers import RMSprop, Adam, SGD # Optimizers for optimizing the model
from tensorflow.keras.callbacks import EarlyStopping  # Regularization method to prevent the overfitting
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import losses, optimizers
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from sklearn import metrics


DATADIR = os.path.abspath(os.getcwd()) + '/data/pets'
CATEGORY = '/cats'
SETS = ['training_set', 'test_set']
IMG_SIZE = 100
CATEGORIES = ['dogs', 'cats']
#path = os.path.join(DATADIR, 'pets' + CATEGORY)    #path to cats or dogs dir
#first_img_path = os.listdir(path)[0]
#img_array = cv2.imread(os.path.join(path, first_img_path), cv2.IMREAD_GRAYSCALE)
#plt.imshow(img_array, cmap = "gray")
#plt.show()
#show image shape
#print('The image shape is {}'.format(img_array.shape))

##########################
# Step 1: Data Preparation
##########################

#create create array of data
data = []

def create_data(i):
    for category in CATEGORIES:
        path = os.path.join(DATADIR, SETS[i], category)  
        class_num = CATEGORIES.index(category)    
        for img in os.listdir(path):
            try: 
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([new_array, class_num])
            except Exception as e:
                print(e)
                pass

create_data(0)
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
#plt.show()

#normalize images
X = X/255.0

######################
# Step 2 Model Traning
######################

#separate training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('the shape of training features is {}'.format(X_train.shape))
print('the shape of training labels is {}'.format(y_train.shape))
print('the shape of test features is {}'.format(X_test.shape))
print('the shape of test labels is {}'.format(y_test.shape))

from tensorflow.keras import backend
backend.clear_session()

# Fixing the seed for random number generators so that we can ensure we receive the same output everytime
np.random.seed(42)
import random
random.seed(42)
tf.random.set_seed(42)

#create model using CNN
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
#model.compile(loss="sparse_categorical_crossentropy",
#             optimizer="adam",
#             metrics=['accuracy'])


# Using SGD Optimizer
opt = SGD(learning_rate = 0.01, momentum = 0.9)

# Compiling the model
#model.compile(optimizer = opt, loss = 'categorical_crossentropy', optimizer="adam", metrics = ['accuracy'])
model.compile(loss="sparse_categorical_crossentropy",
             optimizer=opt,
             metrics=['accuracy'])

# Generating the summary of the model
model.summary()

#fit model
#history = model.fit(X_train, y_train, epochs=5, validation_split=0.2)
# Fitting the model with 30 epochs and validation_split as 10%
history=model.fit(X_train,
          y_train,
          epochs = 3,
          batch_size= 32, validation_split = 0.10)


# Plotting the training and validation accuracies for each epoch

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#model.evaluate(X_test, (y_test))

#show learning curves
#mean training loss and accuracy measured over each epoch
#mean validation loss and accuracy measured at the end of each epoch
#pd.DataFrame(history.history).plot(figsize=(8,5))
#plt.grid(True)
#plt.gca().set_ylim(0,1) # set the vertical range to [0-1]
#plt.show()

#use predict_classes() to find the class with the highest probability
#predict_x=model.predict(X_test) 
#classes_x=np.argmax(predict_x,axis=1)
#y_pred = model.predict(X_test)
#print("Performance Summary of Sequential Neural Network on test data:")
#show classification report
#print(metrics.classification_report(y_test, y_pred))
#show confusion matrix
#print(metrics.confusion_matrix(y_test, y_pred))

#########################
## Step 3 Saving the model
##########################

