import cv2
import matplotlib.pyplot as plt
from Step1_get_data import img_array

IMG_SIZE = 250

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()

