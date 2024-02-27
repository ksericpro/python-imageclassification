import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pickle

# For Data Exploration, ETL
import numpy as np

# For Data Visualization
import cv2
import matplotlib.pyplot as plt
from random import randrange
import inferencemgr


FILENAME = 'models/pets/finalized_model2.sav'
DATADIR = os.path.abspath(os.getcwd()) + '/data/pets/test_set'            # Path of training data after unzipping
CATEGORIES = ['dogs', 'cats']                                                  # Storing all the categories in categories variable
IMG_SIZE = 150                                                                 # Defining the size of the image to 150
# Create a dictionary
dictionary = {} # Curly braces method
dictionary["dogs"] = 0
dictionary["cats"] = 1


def getImageFile(dict):
    # Predict user defined image from location
    path = os.path.join(DATADIR, CATEGORIES[dict])    #path to cats or dogs dir
    total = len(os.listdir(path))

    index = randrange(total)
    print("Chosen index={}".format(index))
    chosen_img_path = os.listdir(path)[index]
    return os.path.join(path, chosen_img_path)

def getMenu(mgr):
    while True:
        print("Menu:")
        print("1. Get from Dogs Directory")
        print("2. Get from Cats Directory")
        print("3. View Image")
        print("q. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == '1':
            imgfile = getImageFile(dictionary["dogs"])
            print("img file={}".format(imgfile))
            predicted = mgr.predictFromImage(imgfile)
            print("Predicted=>{}".format(predicted))
        elif choice == '2':
            imgfile = getImageFile(dictionary["cats"])
            print("img file={}".format(imgfile))
            predicted = mgr.predictFromImage(imgfile)
            #print("Predicted=>{}".format(predicted))
        elif choice == '3':
            mgr.show(imgfile)
        elif choice == 'q':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please enter a valid option ")

def main():
    mgr = inferencemgr.InferenceMgr(FILENAME, IMG_SIZE)
    mgr.loadModel()
    getMenu(mgr)
    mgr.close() 

if __name__ == "__main__":
    main()
