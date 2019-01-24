#!/usr/bin/env python2
from __future__ import print_function
import robobo
import cv2
import sys
import signal
import time
import os

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential, model_from_json, load_model
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

from IPython.display import display
from PIL import Image

import numpy as np
from keras.preprocessing import image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Initialize the CNN
classifier = Sequential()

# Step 1 - Convolution
#number of filters (32), shape for each filter (3, 3), input shape (64, 64), type of image(RGB(3) or B/W), activation function
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = "relu"))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(activation = "relu", units = 128)) #output_dim = 128
classifier.add(Dense(activation = "softmax", units = 6)) #output_dim = 6

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the images
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    './src/images/HW_dataset/training_set',
    target_size = (64, 64),
    batch_size = 32, #Number of observations per batch
    class_mode = 'categorical'
)

test_set = test_datagen.flow_from_directory(
    './src/images/HW_dataset/test_set',
    target_size = (64, 64),
    batch_size = 32, #Number of observations per batch
    class_mode = 'categorical'
)


if __name__ == "__main__":

    # load json and create model
    json_file = open('./src/CNN_HW_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("./src/CNN_HW_weights.h5")
    print("Loaded model from disk")
    loaded_model.save('model_HW_num.hdf5')
    loaded_model = load_model('model_HW_num.hdf5')

    rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.8")

    rob.set_phone_tilt(109, 100)

    time.sleep(0.1)

    def actionStraightForward():
        rob.move(20, 20, 2000)
        time.sleep(0.1)
        print("Forward")

    def action45Right():
        rob.move(20, -5, 1000)
        time.sleep(0.1)
        print("45Right")

    def action90Right():
        rob.move(50, -5, 1000)
        time.sleep(0.1)
        print("90Right")

    def action45Left():
        rob.move(-5, 20, 1000)
        time.sleep(0.1)
        print("45Left")

    def action90Left():
        rob.move(-5, 50, 1000)
        time.sleep(0.1)
        print("90Left")

    def actionBackwards():
        rob.move(-15, -15, 2000)
        time.sleep(0.1)
        print("Backward")

    # images = []

    for i in range(200):

        predict_image = rob.get_image_front()
        # print(predict_image)

        # images.append(predict_image)
        # for i, predict_image in enumerate(images):
        #     cv2.imwrite('./src/images/run/img-' + str(i) + ".png", image)

        cv2.imwrite('./src/images/run/img-0.png', predict_image)
        predict_image = cv2.imread('./src/images/run/img-0.png')
        # print(predict_image)

        # image = rob.get_image_front().reshape([-1, 64, 64, 3])
        # print(image)
        # image = image.load_img(image, target_size=(64, 64))

        predict_image = cv2.resize(predict_image, (64, 64))
        predict_image = predict_image[..., ::-1].astype(np.float32) / 255.0
        predict_image = image.img_to_array(predict_image)
        predict_image = np.expand_dims(predict_image, axis=0) # Add fourth dimension

        output = loaded_model.predict_classes(predict_image)
        print("Predicted output:" + str(output))

        training_set.class_indices
        if output[0] == 0:
            prediction = 'forward'
            actionStraightForward()
        elif output[0] == 1:
            prediction = '45right'
            action45Right()
        elif output[0] == 2:
            prediction = '90right'
            action90Right()
        elif output[0] == 3:
            prediction = '45left'
            action45Left()
        elif output[0] == 4:
            prediction = '90left'
            action90Left()
        elif output[0] == 5:
            prediction = 'backward'
            actionBackwards()

        try:
            os.remove('./src/images/run/img-0.png')
        except:
            pass