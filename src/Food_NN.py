#!/usr/bin/env python2
from __future__ import print_function
import robobo
import cv2
import sys
import signal
import time
import numpy as np


if __name__ == "__main__":

    rob = robobo.SimulationRobobo().connect(address='172.20.10.2', port=19999)
    # rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.16")

    rob.play_simulation()
    # rob.set_phone_pan(343, 100)
    # rob.set_phone_tilt(109, 100)
    # rob.set_phone_pan(11, 100)
    # rob.set_phone_tilt(26, 100)

    time.sleep(0.1)

    x = np.array((rob.read_irs()), dtype=float)
    print("input x:" + str(x))
    print("Begin IRS:" + str(rob.read_irs()))
    if np.amax(x, axis=0) != 0:
        x = x / np.amax(x, axis=0)

    class Neural_Network(object):
        def __init__(self):
            # parameters
            self.inputSize = 8
            self.outputSize = 1
            self.hiddenSize = 5

            # weight matrix from input to hidden layer
            self.W1 = [
                # 10000 runs, 36 data points, 5 hidden nodes
                [-3.0293387728231327, - 6.6555223599347055, 1.3895854668586614, 0.6855186826623935, - 8.736955749205126],
                [- 0.9476957113266768, 7.668494276419445, - 0.7756409281619986, - 5.85312485132007, 0.25389602049900184],
                [0.8340066365985902, 4.659296270881088, 1.1990873969585283, - 2.677505497806095, - 0.15468326910182723],
                [- 1.598148715339809, - 2.5563229287603777, 1.4992281416978608, 1.348813447726324, - 1.500285219083036],
                [- 1.988141278050739, 2.6773660474162697, - 2.8807050972764805, - 2.6505088031395276, 4.745598997810229],
                [0.7819946716042375, 0.7308026519985253, 0.6452142776557535, - 1.0138534210771606, 4.047959517477389],
                [- 2.4681249872590607, 7.657084404350637, - 5.469860669901857, - 1.96862747401349, 4.418101806763281],
                [- 3.4538099829837674, - 5.896433777796302, - 2.141893493879237, 0.7941159461902955, 4.2268310061955905]
            ]

            # weight matrix from hidden to output layer
            self.W2 = [
                # 10000 runs, 36 data points, 5 hidden nodes
                [-2.219847820545832],
                [- 4.923262129595543],
                [3.6166335812156816],
                [- 4.381700423182979],
                [4.593004168524809]
            ]

        def forward(self, x):
            # forward propagation through our network
            self.z = np.dot(x, self.W1)  # dot product of x (input) and first set of 8x3 weights
            self.z2 = self.sigmoid(self.z)  # activation function
            self.z3 = np.dot(self.z2, self.W2)  # dot product of hidden layer (z2) and second set of weights
            o = self.sigmoid(self.z3)  # final activation function
            return o

        def sigmoid(self, s):
            # activation function
            return 1 / (1 + np.exp(-s))


    def actionStraightForward():
        rob.move(20, 20, 600)
        # rob.move(20, 20, 1000)
        print("StraightForward")

    def action45Right():
        rob.move(10, -5, 600)  # for second training weights
        # rob.move(10, -5, 1000) # for third training weights
        print("45Right")

    def action90Right():
        rob.move(10, -10, 600)  # for second training weights
        # rob.move(10, -10, 1000) # for third training weights
        print("90Right")

    def action45Left():
        rob.move(-5, 10, 600)  # for second training weights
        # rob.move(-5, 10, 1000) # for third training weights
        print("45Left")

    def action90Left():
        rob.move(-10, 10, 600)  # for second training weights
        # rob.move(-10, 10, 1000) # for third training weights
        print("90Left")

    def actionBackwards():
        rob.move(-10, -10, 1000)
        print("Backwards")

    # def CameraView():
    #     rob.get_image_front()
    #     #print("Image")

    # def CameraView2(self):
    #     rob._get_image(self._FrontalCamera)

    image = rob.get_image_front()
    cv2.imwrite("test_pictures.png",image)

    for i in range(200):

        #image = rob.get_image_front()
        # image = rob.get_image_front()
        # cv2.imwrite("test_pictures.png", image)

        # Scaling IR signal
        x = np.array((rob.read_irs()), dtype=float)
        if np.amax(x, axis=0) != 0:
            x = x / np.amax(x, axis=0)

        NN = Neural_Network()
        o = NN.forward(x)

        Neural_Network()

        if 0 <= o < 0.1667:
            actionStraightForward()
            time.sleep(0.1)

        elif 0.1667 <= o < 0.3334:
            action45Right()
            time.sleep(0.1)

        elif 0.3334 <= o < 0.5:
            action90Right()
            time.sleep(0.1)

        elif 0.5 <= o < 0.6668:
            action45Left()
            time.sleep(0.1)

        elif 0.6668 <= o <= 0.8335:
            action90Left()
            time.sleep(0.1)

        elif 0.8335 <= o <= 1:
            actionBackwards()
            time.sleep(0.1)

        print("Current IRS: \n" + str(rob.read_irs()))
        print("Predicted Output: \n" + str(o))
