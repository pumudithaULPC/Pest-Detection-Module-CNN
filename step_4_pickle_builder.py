from step_3_building_training_data import *
import pickle
import numpy as np

X = []
y = []
num_of_classes=3

for features,label in training_data:
    ax=np.zeros(num_of_classes)
    ax[label]=1
    X.append(features)
    y.append(ax)


X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()


pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)