import os
from random import randrange

class UtilsMgr:
    def __init__(self):
        print("UtilsMgr")

    def deleteFile(self, filename):
        print("Deleting {}".format(filename))
        os.unlink(filename)

    def getImageFile(self, DATADIR, CATEGORIES, dict):
        # Predict user defined image from location
        path = os.path.join(DATADIR, CATEGORIES[dict])    #path to cats or dogs dir
        total = len(os.listdir(path))

        index = randrange(total)
        print("Chosen index={}".format(index))
        chosen_img_path = os.listdir(path)[index]
        return os.path.join(path, chosen_img_path)
