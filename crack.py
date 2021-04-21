# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 19:13:06 2021

@author: LENOVO
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np

DATADIR = "../input/surface-crack-detection"
training_data = []
categories = ['Negative', 'Positive']
img_size = 227
def creating_training_data():
    for category in categories:
        path = os.path.join(DATADIR, category) 
        class_num = categories.index(category)
        for img in tqdm(os.listdir(path)):  
            
            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  
            new_array = cv2.resize(img_array, (img_size, img_size))  
            training_data.append([new_array, class_num])  
            
            
creating_training_data()     


training_data = random.shuffle(training_data) 

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 1)     

X = np.array(X)/255.0
y = np.array(y)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=60, validation_split=0.3)
