from typing import Union
from fastapi import FastAPI, File, UploadFile
import os
import inferencemgr
from random import randrange

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

FILENAME = 'models/pets/finalized_model2.sav'
DATADIR = os.path.abspath(os.getcwd()) + '/data/pets/test_set'            # Path of training data after unzipping
CATEGORIES = ['dogs', 'cats']                                                  # Storing all the categories in categories variable
IMG_SIZE = 150                                                                 # Defining the size of the image to 150
# Create a dictionary
dictionary = {} # Curly braces method
dictionary["dogs"] = 0
dictionary["cats"] = 1
UPLOAD_DIR = os.path.abspath(os.getcwd()) + '/upload' 

mgr = inferencemgr.InferenceMgr(FILENAME, IMG_SIZE)
mgr.loadModel()

def getImageFile(dict):
    # Predict user defined image from location
    path = os.path.join(DATADIR, CATEGORIES[dict])    #path to cats or dogs dir
    total = len(os.listdir(path))

    index = randrange(total)
    print("Chosen index={}".format(index))
    chosen_img_path = os.listdir(path)[index]
    return os.path.join(path, chosen_img_path)


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to greatest API in the world"}

@app.get("/ping")
def ping():
    return {"message": "pong"}

@app.post("/inference")
def inference():
    imgfile = getImageFile(randrange(2))
    print("img file={}".format(imgfile))
    predicted = mgr.predictFromImage(imgfile)
    print("Predicted=>{}".format(predicted))
    return {"message":"ok", "result":predicted}

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        with open(os.path.join(UPLOAD_DIR, file.filename), 'wb') as f:
            while contents := file.file.read(1024 * 1024):
                f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return {"message": f"Successfully uploaded {file.filename}"}