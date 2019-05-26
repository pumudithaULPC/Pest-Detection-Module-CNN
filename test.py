from pathlib import Path
import tensorflow as tf
import cv2
import numpy as np
CATEGORIES = ["1BrownPlantHopper",  "2StemBorer","3RiceBug",]
#normal_data
path_in_str_in_str ='/home/pumma/Desktop/test_dataset_approach_1'
#complex_data
#path_in_str_in_str ='/home/pumma/Desktop/test_data_complex/data_compex'
pathlist = Path(path_in_str_in_str).glob('**/*.jpg')


model = tf.keras.models.load_model('keras_model.model')


def prepare(path_in_str):
    IMG_SIZE = 250  # 50 in txt-based
    img_array = cv2.imread(path_in_str, cv2.IMREAD_GRAYSCALE)/255.0
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


for path in pathlist:
    # because path is object not string
    path_in_str = str(path)
    print(path_in_str)
    test = model.predict([prepare(path_in_str)])
    max=np.argmax(test)
    print(CATEGORIES[max])
    # print(path_in_str)
    # print(max)  # will be a list in a list.
    # print(CATEGORIES[int(prediction)])