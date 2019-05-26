from keras.layers import Dense,Conv2D,Activation,MaxPooling2D,Dropout,Flatten
from keras.models import Sequential,load_model
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

x_train= pickle.load(open("X.pickle","rb"))/255.0
y_train= pickle.load(open("y.pickle","rb"))

model=Sequential()
model.add(Conv2D(64,(3,3),input_shape=x_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2)))


model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(3))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

model.fit(x_train,y_train,batch_size=50,epochs=10)

model.save('keras_model.model')
