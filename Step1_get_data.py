
import matplotlib.pyplot as plt
import os
import cv2



DATADIR = "/home/pumma/Desktop/fyp_data/data_approach_01/dataa/data"
CATEGORIES = ["1BrownPlantHopper",  "2StemBorer","3RiceBug"]

for category in CATEGORIES:
    path = os.path.join(DATADIR,category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
       # plt.imshow(img_array, cmap='gray')
        print(img_array)

        break
    break
