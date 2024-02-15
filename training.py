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

# Model Saving
import pickle

# Training Data
DATADIR = os.path.abspath(os.getcwd()) + '/data/pets/training_set'            # Path of training data after unzipping
CATEGORIES = ['dogs', 'cats']                                                 # Storing all the categories in 'CATEGORIES' variable
IMG_SIZE = 150 

# Here we will be using a user defined function create_training_data() to extract the images from the directory
training_data = []

# Storing all the training images
def create_training_data():
    for category in CATEGORIES:                                                # Looping over each category from the CATEGORIES list
        path = os.path.join(DATADIR, category)                                 # Joining images with labels
        class_num = category

        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))                    # Reading the data

            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))            # Resizing the images

            training_data.append([new_array, class_num])                       # Appending both the images and labels
    print("{} training record(s)".format(len(training_data)))

create_training_data()


# Testing Data
DATADIR_test = os.path.abspath(os.getcwd()) + '/data/pets/test_set'            # Path of training data after unzipping
CATEGORIES = ['dogs', 'cats']                                                  # Storing all the categories in categories variable
IMG_SIZE = 150                                                                 # Defining the size of the image to 150

# Here we will be using a user defined function create_testing_data() to extract the images from the directory
testing_data = []

# Storing all the testing images
def create_testing_data():
    for category in CATEGORIES:                                                # Looping over each category from the CATEGORIES list
        path = os.path.join(DATADIR_test, category)                            # Joining images with labels
        class_num = category

        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))                    # Reading the data

            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))            # Resizing the images

            testing_data.append([new_array, class_num])                        # Appending both the images and labels
    print("{} testing record(s)".format(len(testing_data)))

create_testing_data()

# Visualize dogs
VISUAL = False
if VISUAL:
    dogs_imgs = [fn for fn in os.listdir(f'{DATADIR}/{CATEGORIES[0]}') ]
    select_dogs = np.random.choice(dogs_imgs, 9, replace = False)

    fig = plt.figure(figsize = (10, 10))

    for i in range(9):
        ax = fig.add_subplot(4, 3, i + 1)

        fp = f'{DATADIR}/{CATEGORIES[0]}/{select_dogs[i]}'

        fn = load_img(fp, target_size = (150, 150))

        plt.imshow(fn, cmap = 'Greys_r')

        plt.axis('off')

    plt.show()

    # cats
    cats_imgs = [fn for fn in os.listdir(f'{DATADIR}/{CATEGORIES[1]}') ]
    select_cats = np.random.choice(cats_imgs, 9, replace = False)

    fig = plt.figure(figsize = (10, 10))

    for i in range(9):
        ax = fig.add_subplot(4, 3, i + 1)

        fp = f'{DATADIR}/{CATEGORIES[1]}/{select_cats[i]}'

        fn = load_img(fp, target_size = (150, 150))

        plt.imshow(fn, cmap = 'Greys_r')

        plt.axis('off')

    plt.show()

# Data Pre-processing
print(testing_data[:3])
# Creating two different lists to store the Numpy arrays and the corresponding labels
X_train = []
y_train = []

np.random.shuffle(training_data)                                               # Shuffling data to reduce variance and making sure that model remains general and overfit less
for features, label in training_data:                                          # Iterating over the training data which is generated from the create_training_data() function
    X_train.append(features)                                                   # Appending images into X_train
    y_train.append(label)                                                      # Appending labels into y_train

print(y_train[:3])

# Creating two different lists to store the Numpy arrays and the corresponding labels
X_test = []
y_test = []

np.random.shuffle(testing_data)                                                # Shuffling data to reduce variance and making sure that model remains general and overfit less
for features, label in testing_data:                                           # Iterating over the training data which is generated from the create_testing_data() function
    X_test.append(features)                                                    # Appending images into X_test
    y_test.append(label)                                                       # Appending labels into y_test

# Converting the pixel values into Numpy array
X_train = np.array(X_train)
X_test = np.array(X_test)
X_train.shape

# Converting the lists into DataFrames
y_train = pd.DataFrame(y_train, columns = ["Label"], dtype = object)
y_test = pd.DataFrame(y_test, columns = ["Label"], dtype = object)


# Checking distribution of classes
# Printing the value counts of target variable
count = y_train.Label.value_counts()
print(count)

print('*'*10)

count = y_train.Label.value_counts(normalize = True)
print(count)

# Normalize the data
X_train_prev = X_train.copy()
X_test_prec = X_test.copy()
print(X_train_prev[0:3])

# Normalizing the image data
X_train = X_train/255.0
X_test = X_test/255.0
print(X_train[0:3])

# Encoding Target Variable
y_train_encoded = [ ]

for label_name in y_train["Label"]:
    if(label_name == 'dogs'):
        y_train_encoded.append(0)

    if(label_name == 'cats'):
        y_train_encoded.append(1)

y_train_encoded = to_categorical(y_train_encoded, 3)
print(y_train_encoded)

y_test_encoded = [ ]

for label_name in y_test["Label"]:
    if(label_name == 'dogs'):
        y_test_encoded.append(0)

    if(label_name == 'cats'):
        y_test_encoded.append(1)

y_test_encoded = to_categorical(y_test_encoded, 3)
print(y_test_encoded)

#################
## Model Building
################

from tensorflow.keras import backend
backend.clear_session()

# Fixing the seed for random number generators so that we can ensure we receive the same output everytime
np.random.seed(42)
import random
random.seed(42)
tf.random.set_seed(42)

# Intializing a sequential model
model = Sequential()

# Adding first conv layer with 64 filters and kernel size 3x3, padding 'same' provides the output size same as the input size
# The input_shape denotes input image dimension
model.add(Conv2D(64, (3, 3), activation = 'relu', padding = "same", input_shape = (150, 150, 3)))

# Adding max pooling to reduce the size of output of first conv layer
model.add(MaxPooling2D((2, 2), padding = 'same'))

# Adding second conv layer with 32 filters and kernel size 3x3, padding 'same' followed by a Maxpooling2D layer
model.add(Conv2D(32, (3, 3), activation = 'relu', padding = "same"))
model.add(MaxPooling2D((2, 2), padding = 'same'))

# Add third conv layer with 32 filters and kernel size 3x3, padding 'same' followed by a Maxpooling2D layer
model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
model.add(MaxPooling2D((2, 2), padding = 'same'))

# Flattening the output of the conv layer after max pooling to make it ready for creating dense connections
model.add(Flatten())

# Adding a fully connected dense layer with 100 neurons
model.add(Dense(100, activation = 'relu'))

# Adding the output layer with 3 neurons and activation functions as softmax since this is a multi-class classification problem
model.add(Dense(3, activation = 'softmax'))

# Using SGD Optimizer
opt = SGD(learning_rate = 0.01, momentum = 0.9)

# Compiling the model
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Generating the summary of the model
model.summary()


##############
## Traing Model
##############

# The following lines of code saves the best model's parameters if training accuracy goes down on further training
es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 5)
mc = ModelCheckpoint('best_model.h5', monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only = True)

# Fitting the model with 30 epochs and validation_split as 10%
history=model.fit(X_train,
          y_train_encoded,
          epochs = 60,
          batch_size= 32, validation_split = 0.10, callbacks = [es, mc])

# save the model to disk
filename = 'models/pets/finalized_model1.sav'
pickle.dump(model, open(filename, 'wb'))

# Plotting the training and validation accuracies for each epoch
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

model.evaluate(X_test, (y_test_encoded))

# Plotting Confusion Matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

pred = model.predict(X_test)
pred = np.argmax(pred, axis = 1)
y_true = np.argmax(y_test_encoded, axis = 1)

# Printing the classification report
print(classification_report(y_true, pred))

# Plotting the heatmap using confusion matrix
cm = confusion_matrix(y_true, pred)
plt.figure(figsize = (8, 5))
sns.heatmap(cm, annot = True,  fmt = '.0f', xticklabels = ['Dogs', 'Cats'], yticklabels=['Dogs', 'Cats'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Model2
from tensorflow.keras import backend
backend.clear_session()

# Fixing the seed for random number generators so that we can ensure we receive the same output everytime
np.random.seed(42)
import random
random.seed(42)
tf.random.set_seed(42)

# Initializing a sequential model
model_2 = Sequential()

# Adding first conv layer with 256 filters and kernel size 5x5, with ReLU activation and padding 'same' provides the output size same as the input size
# The input_shape denotes input image dimension
model_2.add(Conv2D(filters = 256, kernel_size = (5, 5), padding = 'Same', activation = 'relu', input_shape = (150, 150, 3)))

# Adding max pooling to reduce the size of output of first conv layer
model_2.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

#  Adding dropout to randomly switch off 25% neurons to reduce overfitting
model_2.add(Dropout(0.25))

# Adding second conv layer with 128 filters and with kernel size 5x5 and ReLu activation function
model_2.add(Conv2D(filters = 128, kernel_size = (5, 5), padding = 'Same', activation = 'relu'))

# Adding max pooling to reduce the size of output of first conv layer
model_2.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

#  Adding dropout to randomly switch off 25% neurons to reduce overfitting
model_2.add(Dropout(0.25))

# Adding third conv layer with 64 filters and with kernel size 3x3 and ReLu activation function
model_2.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'Same', activation = 'relu'))

# Adding max pooling to reduce the size of output of first conv layer
model_2.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

#  Adding dropout to randomly switch off 25% neurons to reduce overfitting
model_2.add(Dropout(0.25))

# Adding fourth conv layer with 32 filters and with kernel size 3x3 and ReLu activation function
model_2.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'Same', activation = 'relu'))

# Adding max pooling to reduce the size of output of first conv layer
model_2.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

#  Adding dropout to randomly switch off 25% neurons to reduce overfitting
model_2.add(Dropout(0.25))

# Flattening the 3-d output of the conv layer after max pooling to make it ready for creating dense connections
model_2.add(Flatten())

# Adding first fully connected dense layer with 64 neurons
model_2.add(Dense(64, activation = "relu"))

# Adding second fully connected dense layer with 32 neurons
model_2.add(Dense(32, activation = "relu"))

# Adding the output layer with 3 neurons and activation functions as softmax since this is a multi-class classification problem
model_2.add(Dense(3, activation = "softmax"))

# Using Adam Optimizer
optimizer = Adam(lr = 0.001)

# Compile the model
model_2.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics = ["accuracy"])

model_2.summary()


# Training 
es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 5)
mc = ModelCheckpoint('best_model.h5', monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only = True)

history=model_2.fit(X_train,
          y_train_encoded,
          epochs = 20,
          batch_size = 16, validation_split = 0.20, use_multiprocessing = True)

# save the model to disk
filename = 'models/pets/finalized_model2.sav'
pickle.dump(model, open(filename, 'wb'))

# Check Accuracy
model_2.evaluate(X_test, y_test_encoded)


# Plotting Confusion Matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

pred = model_2.predict(X_test)
pred = np.argmax(pred, axis = 1)
y_true = np.argmax(y_test_encoded, axis = 1)

#Printing the classification report
print(classification_report(y_true, pred))

#Plotting the heatmap using confusion matrix
cm = confusion_matrix(y_true, pred)
plt.figure(figsize = (8, 5))
sns.heatmap(cm, annot = True,  fmt = '.0f', xticklabels = ['Dogs', 'Cats'], yticklabels = ['Dogs', 'Cats'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
