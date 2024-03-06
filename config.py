import os
MODEL="models/pets/finalized_model2.sav"
CATEGORIES=['dogs', 'cats']                                                 
IMG_SIZE=150
DATADIR = os.path.abspath(os.getcwd()) + '/data/pets/test_set'
UPLOAD_DIR = os.path.abspath(os.getcwd()) + '/upload' 