import cv2
import numpy as np
import tensorflow as tf


CATEGORIES = ["1BrownPlantHopper",  "2StemBorer","3RiceBug"]


def prepare(filepath):
    IMG_SIZE =   250# 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)/255.0
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model('keras_model.model')



test = model.predict(prepare('/home/pumma/Downloads/test_noise.jpg'))
print(test)
max=np.argmax(test)
print(CATEGORIES[max])

