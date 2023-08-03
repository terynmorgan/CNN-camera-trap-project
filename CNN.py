# Author: Teryn Morgan
import numpy as np 
import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.preprocessing import LabelEncoder 
from keras import backend as K
from keras_preprocessing.image import load_img
from keras.utils import to_categorical
import random
import os
from sklearn.model_selection import train_test_split
import cv2
from keras.utils import np_utils
import random

# import data
path = 'Project/lila_downloads_by_species/wellington-unzipped/images'
dir_list = os.listdir(path)
dir_list = [file.lower() for file in dir_list]

wellington_df = pd.read_csv('Project/wellington_camera_traps.csv', dtype=str)
wellington_df['file'] = wellington_df['file'].apply(lambda x: x.lower())

# Encoding of y labels (17)
label_y = wellington_df['label']
encoder = LabelEncoder()
encoder.fit(label_y)
encoded_y = encoder.transform(label_y)
encoded_y = np_utils.to_categorical(encoded_y, 17)

# Resize images 3264 x 2488 -> 256 x 256 
IMG_SIZE = 256

# Preprocess image data into pixel data 
train_data = []

for image in dir_list:
    # Get label for image from encoded_y
    label = encoded_y[dir_list.index(image)]

    # Load and resize image 
    img_array = cv2.imread(os.path.join(path, image))
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

    train_data.append([new_array, label])

random.shuffle(train_data)

# Assign features and labels
X =[]
y =[]
for features, label in train_data:
  X.append(features)
  y.append(label)

X = np.array(X)
y = np.array(y)

X.shape #(270450 , 256, 256, 3) 
y.shape #(270450 , 17)

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D((2, 2), strides=2))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(17, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, batch_size = 32, epochs = 16, verbose = 1, validation_data = (X_test, y_test))

loss, acc = model.evaluate(X_test, y_test, verbose = 0)

# Save the model 
model.save_weights('model_weights.h5')
model.save('model_keras.h5')