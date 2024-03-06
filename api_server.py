from typing import Union
from fastapi import FastAPI, File, UploadFile
import os
import inferencemgr
import utilsmgr
from random import randrange
from fastapi.middleware.cors import CORSMiddleware
from os.path import join, dirname
from dotenv import load_dotenv
import config

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)
MODEL = config.MODEL                                             
IMG_SIZE = config.IMG_SIZE
CATEGORIES=config.CATEGORIES                                              
print("MODEL={}, CATEGORIES={}, IMG_SIZE={}".format(MODEL, CATEGORIES, IMG_SIZE))

DATADIR = config.DATADIR
UPLOAD_DIR = config.UPLOAD_DIR

utils_mgr = utilsmgr.UtilsMgr()
inf_mgr = inferencemgr.InferenceMgr(MODEL, IMG_SIZE)
inf_mgr.loadModel()

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to greatest API in the world"}

@app.get("/ping")
def ping():
    return {"message": "pong"}

@app.get("/inference")
def inference():
    imgfile = utils_mgr.getImageFile(DATADIR, CATEGORIES, randrange(2))
    print("img file={}".format(imgfile))
    predicted = inf_mgr.predictFromImage(imgfile)
   # print("Predicted=>{}".format(predicted))
    return {"message":"ok", "result":predicted}

@app.post("/predict")
def upload(file: UploadFile = File(...)):
    filename = os.path.join(UPLOAD_DIR, file.filename)
    print("Saving to {}".format(filename))
    try: 
        with open(filename, 'wb') as f:
            while contents := file.file.read(1024 * 1024):
                f.write(contents)
       
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    predicted = inf_mgr.predictFromImage(filename)
    #print("Predicted=>{}".format(predicted))
    utils_mgr.deleteFile(filename)
    return {"message":"ok", "result":predicted}