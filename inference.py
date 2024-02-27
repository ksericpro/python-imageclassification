import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pickle

# For Data Exploration, ETL
import numpy as np

# For Data Visualization
import cv2
import matplotlib.pyplot as plt

#######################
## Step 1 Load Model
#######################
FILENAME = 'models/pets/finalized_model2.sav'
SHOW = True

print("Loading model {}".format(FILENAME))
# open a file, where you stored the pickled data
file = open(FILENAME, 'rb')

# dump information to that file
model = pickle.load(file)

# close the file
file.close()

####################
## Step 2 Prediction
####################
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
        ct = 0
        for img in os.listdir(path):
            ct = ct + 1
            img_array = cv2.imread(os.path.join(path, img))                    # Reading the data

            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))            # Resizing the images
            #show image shape
            print("#{}. The old image shape is {}. The new image shape is {}".format(ct, img_array.shape, new_array.shape))

            testing_data.append([new_array, class_num])                        # Appending both the images and labels
    print("{} testing record(s) in total.".format(len(testing_data)))

create_testing_data()

# Creating two different lists to store the Numpy arrays and the corresponding labels
X_test = []

np.random.shuffle(testing_data)                                                # Shuffling data to reduce variance and making sure that model remains general and overfit less
for features, label in testing_data:                                           # Iterating over the training data which is generated from the create_testing_data() function
    X_test.append(features)                                                    # Appending images into X_test

X_test = np.array(X_test)

# Predicting the test image with the best model and storing the prediction value in res variable
res = model.predict(X_test[1].reshape(1, 150, 150, 3))

print(res)
# Applying argmax on the prediction to get the highest index value
i=np.argmax(res)
predicted = "Unknown"
if(i == 0):
    predicted = "Dogs"
if(i==1):
    predicted = "Cats"

print("Predicted=>{}".format(predicted))


# Predict user defined image from location
path = os.path.join(DATADIR_test, CATEGORIES[0])    #path to cats or dogs dir
total = len(os.listdir(path))
from random import randrange
index = randrange(total)
print("Chosen index={}".format(index))
chosen_img_path = os.listdir(path)[index]

print("\n[{}] Loading Image file={}/{}".format(total, path, chosen_img_path))
img_array = cv2.imread(os.path.join(path, chosen_img_path))

# predict
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

# show image shape
print("The old image shape is {}. The new image shape is {}".format(img_array.shape, new_array.shape))

new_array_np = np.array(new_array)

# Predicting the test image with the best model and storing the prediction value in res variable
res = model.predict(new_array_np.reshape(1, 150, 150, 3))
print(res)

# Applying argmax on the prediction to get the highest index value
i=np.argmax(res)
predicted = "Unknown"
if(i == 0):
    predicted = "Dogs"
if(i==1):
    predicted = "Cats"

print("Predicted=>{}".format(predicted))

if SHOW:
    plt.title = chosen_img_path
    plt.imshow(img_array, cmap = "gray")
    plt.show()
