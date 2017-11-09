import csv
import cv2

#use python CSV library (csv) to read and store the lines from the driving log csv file
#and then for each line I will extract the path to the camera image
lines = []
with open('./data/driving_log.csv') as csvfile:
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
      current_path='./data/IMG/' + filename
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

#After load the images and the steering measurement, cover them to NumPy array (numpy)
#because Keras requires NumPy array format
    
import numpy as np

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

#I started with implementing simple ConvNetwork- LeNet, afterward, try Nvidia Arch, which is much more powerful network 
model = Sequential ()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160,320,3))) #add Lambda for data preprocessing: normalize the data by dividing each element by 255
model.add(Cropping2D(cropping=((70, 25),(0,0)))) #Crop the images and only leave the necessary section, i.e. road
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

#with the network constructed, i will compile the model
#for the loss funciton I use the mean squared error, or MSE.
model.compile(loss = 'mse', optimizer = 'adam')

#Once the model is compiled, I will train it with the feature and label arrays i just built
#Will also shuffle the data and split off 20% of the data to use for a validation set
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch=7) 

#save the trained model and use it to drive the vehicle in autonomous mode in the simulator
model.save('model.h5')
