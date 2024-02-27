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

# Model Saving
import pickle
import argparse

CATEGORIES = ['dogs', 'cats']
IMG_SIZE = 100 
DATADIR = os.path.abspath(os.getcwd()) + '/data/pets/training_set' 
MODEL = 'models/pets/gradio_model.sav'
model = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", default="0")

    args = parser.parse_args()
    train = args.train
    print("train="+train)

    if train=="1":
        trainandsave()

    # Load Model
    print("Loading model {}".format(MODEL))

    if not(os.path.exists(MODEL)):
        print("model file {} not found.".format(MODEL))
        return
    
    # open a file, where you stored the pickled data
    file = open(MODEL, 'rb')

    # dump information to that file 
    global model
    model = pickle.load(file)

    # close the file
    file.close()

    # Launch UI
    iface.launch(share=True)
    
def trainandsave():
    print("-Training-")
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

    #Modelling
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
    history = model.fit(X_train, y_train, epochs=20, batch_size= 32, validation_split=0.1)

    #show learning curves
    #mean training loss and accuracy measured over each epoch
    #mean validation loss and accuracy measured at the end of each epoch
    #pd.DataFrame(history.history).plot(figsize=(8,5))
    #plt.grid(True)
    #plt.gca().set_ylim(0,1) # set the vertical range to [0-1]
    #plt.show()

    #use predict_classes() to find the class with the highest probability
    #y_pred = model.predict_classes(X_test)
    y_pred = model.predict(X_test)
    print("Performance Summary of Sequential Neural Network on test data:")

    # save the model to disk
    pickle.dump(model, open(MODEL, 'wb'))


#create a function to make predictions
#return a dictionary of labels and probabilities
def cat_or_dog(img):
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    prediction = model.predict(img).tolist()[0]
    print("predicion=>".format(prediction))
    class_names = ["Dog", "Cat"]
    return {class_names[i]: prediction[i] for i in range(2)}
#set the user uploaded image as the input array
#match same shape as the input shape in the model
im = gradio.inputs.Image(shape=(IMG_SIZE, IMG_SIZE), image_mode='L', invert_colors=False, source="upload")
#setup the interface
iface = gr.Interface(
    fn = cat_or_dog, 
    inputs = im, 
    outputs = gradio.outputs.Label(),
)

if __name__ == "__main__":
    main()