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

if __name__ == "__main__":

    # rob = robobo.SimulationRobobo().connect(address='130.37.245.223', port=19997)
    rob = robobo.HardwareRobobo(camera=True).connect(address="172.20.10.11")

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

    images = []

    for i in range(10):

        image = rob.get_image_front()
        # print(image)

        images.append(image)
        for i, image in enumerate(images):
            cv2.imwrite('./src/images/HW_dataset/HW_p1-' + str(i) + ".png", image)


