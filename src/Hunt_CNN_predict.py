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

    # Checkpoint models
    # loaded_model = load_model('./src/week3/models/CNN_Sim_weights(10)01-0.56.hdf5')
    # loaded_model = load_model('./src/week3/models/CNN_Sim_weights(10)02-0.48.hdf5')
    # loaded_model = load_model('./src/week3/models/CNN_Sim_weights(10)03-0.47.hdf5')
    # loaded_model = load_model('./src/week3/models/CNN_Sim_weights(10)04-0.80.hdf5')
    # loaded_model = load_model('./src/week3/models/CNN_Sim_weights(10)05-0.55.hdf5')
    # loaded_model = load_model('./src/week3/models/CNN_Sim_weights(10)06-0.76.hdf5')
    # loaded_model = load_model('./src/week3/models/CNN_Sim_weights(10)07-0.85.hdf5')
    # loaded_model = load_model('./src/week3/models/CNN_Sim_weights(10)08-1.22.hdf5')
    # loaded_model = load_model('./src/week3/models/CNN_Sim_weights(10)09-0.98.hdf5')
    loaded_model = load_model('./src/week4/models/CNN_Sim_weights(2)08-0.81.hdf5')

    signal.signal(signal.SIGINT, terminate_program)

    rob = robobo.SimulationRobobo().connect(address='192.168.1.14', port=19997)

    rob.play_simulation()

    # connect to prey robot
    prey_robot = robobo.SimulationRoboboPrey().connect(address='192.168.1.14', port=19989)
    # initialise class prey
    prey_controller = prey.Prey(robot=prey_robot, level=4)
    # start the thread prey, makes the prey move
    prey_controller.start()

    # rob.set_phone_pan(343, 100)
    # rob.set_phone_tilt(109, 100)
    rob.set_phone_tilt(32, 100)

    # plat op zijn rug
    # rob.set_phone_pan(11, 100)
    # rob.set_phone_tilt(26, 100)

    time.sleep(1)

    def actionStraightForward():
        rob.move(25, 25, 1000)
        time.sleep(0.1)
        print("StraightForward")

    def action20Right():
        rob.move(5, -2, 500)
        time.sleep(0.1)
        print("20Right")

    def action45Right():
        rob.move(5, -5, 400)
        time.sleep(0.1)
        print("45Right")

    def action90Right():
        rob.move(20, -10, 600)
        time.sleep(0.1)
        print("90Right")

    def action20Left():
        rob.move(-2, 5, 500)
        time.sleep(0.1)
        print("20Left")

    def action45Left():
        rob.move(-5, 5, 400)
        time.sleep(0.1)
        print("45Left")

    def action90Left():
        rob.move(-10, 20, 600)
        time.sleep(0.1)
        print("90Left")

    def actionBackwards():
        rob.move(-10, -10, 500)
        time.sleep(0.1)
        print("Backwards")

    # images = []

    for i in range(100):

        predict_image = rob.get_image_front()
        # print(predict_image)

        # Use this to save images for database
        # images.append(predict_image)
        # for i, predict_image in enumerate(images):
        #     cv2.imwrite('./src/week3/images/run/img-' + str(i) + ".png", predict_image)

        # temporarily save image to computer and load image
        # cv2.imwrite('./src/week4/images/run/img-0.png', predict_image)
        # predict_image = cv2.imread('./src/week4/images/run/img-0.png')

        # predict_image = cv2.imread(predict_image)

        # print(predict_image)
        predict_image = cv2.resize(predict_image, (64, 64))
        predict_image = predict_image[..., ::-1].astype(np.float32) / 255.0
        predict_image = image.img_to_array(predict_image)
        predict_image = np.expand_dims(predict_image, axis=0)  # Add fourth dimension

        # predicted output
        output = loaded_model.predict_classes(predict_image)
        print("Predicted output:" + str(output))

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

        # Remove temporarily image from computer
        # try:
        #     os.remove('./src/week4/images/run/img-0.png')
        # except:
        #     pass

    # stop the prey
    prey_controller.stop()
    prey_controller.join()
    prey_robot.disconnect()

    # Stopping the simulation resets the environment
    rob.stop_world()