#!/usr/bin/env python2
from __future__ import print_function
import robobo
import prey
from cv2 import *
import cv2
import sys
import signal
import time
import os
import datetime

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


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)

if __name__ == "__main__":

    # load json and create model
    # json_file = open('./src/week3/models/CNN_model(6).json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    # loaded_model.load_weights("./src/week3/modelsCNN_weights(3).h5")
    # print("Loaded model from disk")
    # loaded_model.save('./src/week3/models/model_num.hdf5')
    # loaded_model = load_model('./src/week3/models/model_num.hdf5')

    loaded_model = load_model('./src/week4/models/CNN_HW_weights(2)02-0.00.hdf5')
    print("Loaded model from disk")

    signal.signal(signal.SIGINT, terminate_program)

    rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.21")

    # Connect to prey robot
    # prey_robot = robobo.SimulationRoboboPrey().connect(address='192.168.1.14', port=19989)
    # prey_robot = robobo.HardwareRobobo().connect(address="192.168.1.7")
    # initialise class prey
    # prey_controller = prey.Prey(robot=prey_robot, level=4)
    # start the thread prey, makes the prey move
    # prey_controller.start()

    # Real world
    # rob.set_phone_pan(343, 100)
    rob.set_phone_tilt(109, 100)

    time.sleep(0.1)

    def actionStraightForward():
        rob.move(50, 50, 1000)
        # rob.move(20, 20, 2000)
        # time.sleep(0.1)
        print("StraightForward")

    def action20Right():
        rob.move(10, -5, 1000)
        # rob.move(20, -5, 1000)
        # time.sleep(0.1)
        print("20Right")

    def action90Right():
        rob.move(50, -5, 1000)
        # time.sleep(0.1)
        print("90Right")

    def action20Left():
        rob.move(-5, 10, 1000)
        # rob.move(-5, 20, 1000)
        # time.sleep(0.1)
        print("20Left")

    def action90Left():
        rob.move(-5, 50, 1000)
        # time.sleep(0.1)
        print("90Left")

    def actionBackwards():
        rob.move(-20, -20, 1000)
        # rob.move(-15, -15, 2000)
        # time.sleep(0.1)
        print("Backwards")


    i = 0

    for i in range(100):

        # start_time = time.time()
        predict_image = rob.get_image_front()[100:]
        # print(predict_image.size)

        # Use this to save images for database
        cv2.imwrite('./src/week4/images/HW_dataset/img_p29-{}.png'.format(i), predict_image)
        i = i+1

        predict_image = cv2.resize(predict_image, (64, 64))
        predict_image = predict_image[..., ::-1].astype(np.float32) / 255.0
        predict_image = image.img_to_array(predict_image)
        predict_image = np.expand_dims(predict_image, axis=0)  # Add fourth dimension
        # print("Preparing image took:\t{}".format(time.time() - start_time))

        # Predicted output
        # start_time = time.time()
        output = loaded_model.predict_classes(predict_image)
        print("Predicted output:" + str(output))
        # print("Predicting image took:\t{}".format(time.time() - start_time))


        if output[0] == 0:
            prediction = 'straight'
            actionStraightForward()
        elif output[0] == 1:
            prediction = '20right'
            action20Right()
        elif output[0] == 2:
            prediction = '90right'
            action90Right()
        elif output[0] == 3:
            prediction = '20left'
            action20Left()
        elif output[0] == 4:
            prediction = '90left'
            action90Left()
        elif output[0] == 5:
            prediction = 'back'
            actionBackwards()