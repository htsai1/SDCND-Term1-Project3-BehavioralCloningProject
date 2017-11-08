import csv
import cv2

lines = []
with open('./data9/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
     lines.append(line) 

images = []
measurements = []

for line in lines:
    for i in range(3):
      source_path = line[i]
      tokens = source_path.split('\\') 
      filename = tokens[-1]    #file name is the very last token
      current_path='./data9/IMG/' + filename
      image = cv2.imread(current_path)
      images.append(image)
    correction = 0.2 
    measurement = float(line[3]) #from string to float
    measurements.append(measurement)
    measurements.append(measurement+correction) # steer more to right
    measurements.append(measurement-correction) # steer more to left

#try to augment dataset for data balancing 
augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements): #run the loop on them together
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image, 1) #1 is flipped horizontally 
    flipped_measurement = measurement * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)

import numpy as np

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

#start with implementing simple ConvNetwork- LeNet
model = Sequential ()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160,320,3))) #normalize data
model.add(Cropping2D(cropping=((70, 25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2, 2), activation = 'relu'))
model.add(Convolution2D(36,5,5,subsample=(2, 2), activation = 'relu'))
model.add(Convolution2D(48,5,5,subsample=(2, 2), activation = 'relu'))
model.add(Convolution2D(64,3,3, activation = 'relu'))
model.add(Convolution2D(64,3,3, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#compile the model
model.compile(loss = 'mse', optimizer = 'adam')
#train the model
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch=7) 


model.save('model.h5')
