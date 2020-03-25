import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator()



model = Sequential()
model.add(Dense(120, input_dim=510, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(60, activation='relu')
# model.add(Dense(60, activation='relu', activity_regularizer=l1(0.001)))
# model.add(Dense(60, activation='relu', kernel_regularizer=L1L2(l1=0.0, l2=0.1)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.save('./weight.h5')