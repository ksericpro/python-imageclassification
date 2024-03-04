import requests
import os 
from random import randrange

DATADIR = os.path.abspath(os.getcwd()) + '/data/pets/test_set'            # Path of training data after unzipping
CATEGORIES = ['dogs', 'cats']     
url = 'http://127.0.0.1:8000/upload'

path = os.path.join(DATADIR, CATEGORIES[randrange(2)]) 
total = len(os.listdir(path))

index = randrange(total)
print("Chosen index={}".format(index))
chosen_img_path = os.listdir(path)[index]
full_file_path = os.path.join(path, chosen_img_path)
print("img file={}".format(full_file_path))

file = {'file': open(full_file_path, 'rb')}
resp = requests.post(url=url, files=file) 
print(resp.json())