#!/usr/bin/env python2
from __future__ import print_function
import robobo
import cv2
import sys
import signal
import time
import Food_CNN_training

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from keras.models import model_from_json, load_model
from keras.preprocessing.image import ImageDataGenerator

from IPython.display import display
from PIL import Image

import numpy as np
from keras.preprocessing import image


if __name__ == "__main__":

    # load json and create model
    json_file = open('./src/CNN_model128.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("./src/CNN_weights128.h5")
    print("Loaded model from disk")
    loaded_model.save('model_num.hdf5')
    loaded_model = load_model('model_num.hdf5')

    rob = robobo.SimulationRobobo().connect(address='130.37.245.223', port=19997)
    # rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.16")

    rob.play_simulation()

    # rob.set_phone_pan(343, 100)
    # rob.set_phone_tilt(109, 100)
    rob.set_phone_tilt(32, 100)

    # plat op zijn rug
    # rob.set_phone_pan(11, 100)
    # rob.set_phone_tilt(26, 100)

    time.sleep(0.1)

    def actionStraightForward():
        rob.move(20, 20, 400)
        # rob.move(20, 20, 1000)
        time.sleep(0.1)
        print("StraightForward")

    def action45Right():
        rob.move(10, -5, 400)
        time.sleep(0.1)
        print("45Right")

    def action90Right():
        rob.move(10, -10, 400)
        time.sleep(0.1)
        print("90Right")

    def action45Left():
        rob.move(-5, 10, 400)
        time.sleep(0.1)
        print("45Left")

    def action90Left():
        rob.move(-10, 10, 400)
        time.sleep(0.1)
        print("90Left")

    def actionBackwards():
        rob.move(-10, -10, 200)
        time.sleep(0.1)
        print("Backwards")

    # images = []

    for i in range(200):

        # image = rob.get_image_front()
        # images.append(image)
        # for i, image in enumerate(images):
        #     cv2.imwrite('./src/images/img_p2-' + str(i) + ".png", image)

        # image = image.load_img(rob.get_image_front(), target_size=(64, 64))
        image = Image.fromarray(rob.get_image_front(), target_size=(64, 64))
        image = image.img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # image = rob.get_image_front().reshape([-1, 128, 128, 3])

        output = loaded_model.predict_classes(image)

        # output = loaded_model.predict_classes(rob.get_image_front())
        print("Predicted output:" + str(output))

        training_set.class_indices
        if output[0][0] == 1:
            prediction = 'straight'
            actionStraightForward()
        elif output[0][1] == 1:
            prediction = '45right'
            action45Right()
        elif output[0][2] == 1:
            prediction = '90right'
            action90Right()
        elif output[0][3] == 1:
            prediction = '45left'
            action45Left()
        elif output[0][4] == 1:
            prediction = '90left'
            action90Left()
        elif output[0][5] == 1:
            prediction = 'back'
            actionBackwards()

    # pause the simulation and read the collected food
    rob.pause_simulation()
    print("Robobo collected {} food".format(rob.collected_food()))

    # Stopping the simulation resets the environment
    rob.stop_world()