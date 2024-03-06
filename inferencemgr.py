from time import perf_counter as pc
import datetime
import pickle
import os
import datetime
# For Data Exploration, ETL
import numpy as np

# For Data Visualization
import cv2
import matplotlib.pyplot as plt

class InferenceMgr:
    def __init__(self, modelfile, imgsize):
        print("Inference Mgr")
        self.modelfile = modelfile
        self.imgsize = imgsize

    def loadModel(self):
        print("Loading model {}".format(self.modelfile))

        if not(os.path.exists(self.modelfile)):
            print("model file {} not found.".format(self.modelfile))
            return
        
        # open a file, where you stored the pickled data
        file = open(self.modelfile, 'rb')

        # dump information to that file
        self.model = pickle.load(file)

        # close the file
        file.close()

        print("model file {} load successfully.".format(self.modelfile))

    def predictFromImage(self, filename):
        print("Loading Image file={}".format(filename))
        t0 = pc()
       # tm = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")
        img_array = cv2.imread(filename)

        # predict
        new_array = cv2.resize(img_array, (self.imgsize, self.imgsize))

        # show image shape
        print("The old image shape is {}. The new image shape is {}".format(img_array.shape, new_array.shape))

        new_array_np = np.array(new_array)

        # Predicting the test image with the best model and storing the prediction value in res variable
        res = self.model.predict(new_array_np.reshape(1, self.imgsize, self.imgsize, 3))
        print(res)

        # Applying argmax on the prediction to get the highest index value
        i=np.argmax(res)
        predicted = "Unknown"
        if(i == 0):
            predicted = "Dogs"
        if(i==1):
            predicted = "Cats"

        print("Predicted=>{}".format(predicted))
        duration = pc() - t0
        print("duration={}s".format(duration))
        return predicted
    
    def show(self, filename):
        print("Loading Image file={}".format(filename))
        img_array = cv2.imread(filename)
        plt.title = filename
        plt.imshow(img_array)
        plt.show()
    
    def close(self):
        print("\n[close]")