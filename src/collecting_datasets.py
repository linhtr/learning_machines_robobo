#!/usr/bin/env python2
from __future__ import print_function
import robobo
import cv2
import sys
import signal
import time
import prey
import numpy as np


if __name__ == "__main__":

    # connect to Robobo
    # rob = robobo.SimulationRobobo().connect(address='10.107.1.100', port=19997)
    rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.21")

    # rob.play_simulation()

    # connect to prey robot
    # prey_robot = robobo.SimulationRoboboPrey().connect(address='10.107.1.100', port=19989)
    # initialise class prey
    # prey_controller = prey.Prey(robot=prey_robot)
    # start the thread prey, makes the prey move
    # prey_controller.start()

    # rob.set_phone_pan(343, 100)
    rob.set_phone_tilt(109, 100)

    # rob.set_phone_tilt(32, 100)

    # plat op zijn rug
    # rob.set_phone_pan(11, 100)
    # rob.set_phone_tilt(26, 100)

    # time.sleep(0.1)

    class Neural_Network(object):
        def __init__(self):
            # parameters
            self.inputSize = 8
            self.outputSize = 1
            self.hiddenSize = 5

            # weight matrix from input to hidden layer
            self.W1 = [
                # 100000 runs, 36 data points, 5 hidden nodes
                [-8.254917929969324, - 1.9166586162658048, 1.5807866151162706, - 8.456242493711311, - 3.867973366850627],
                [- 0.11216853848288441, 0.3973288667277819, - 2.1391698441249423, - 1.9780521314028934, - 6.235392640025527],
                [7.857855667151483, - 1.4106382855801325, - 2.351902169175414, 1.4580313385819246, 1.229245232081598],
                [- 8.74270677068826, 5.815752183289975, 3.244665484752045, - 26.221136086714093, - 16.798545038973838],
                [2.7293684046642652, - 2.580748626109036, - 5.41486108631858, - 11.710331227548334, - 0.32471896298413505],
                [10.013490204721387, 2.0997053366914553, - 2.7032298429212953, - 11.683174424341088, - 3.3314507252583243],
                [9.412420822045343, - 0.8127493029028986, - 18.669845974654496, - 14.945099263866249, - 9.673198518555274],
                [- 2.679996522937562, - 3.3083923786677234, 9.884323928881969, 25.99069612204657, 17.745350642459613]

            ]

            # weight matrix from hidden to output layer
            self.W2 = [
                # 100000 runs, 36 data points, 5 hidden nodes
                [-3.1509179277733623],
                [5.121131963815984],
                [- 8.23155998143664],
                [- 15.374341304480232],
                [17.590519228430114]
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
        # rob.move(20, 20, 400)
        rob.move(40, 40, 1000)
        print("StraightForward")

    def action45Right():
        # rob.move(10, -5, 400)
        rob.move(5, -5, 1000)
        print("45Right")

    def action90Right():
        rob.move(10, -10, 1000)
        print("90Right")

    def action45Left():
        # rob.move(-5, 10, 400)
        rob.move(-5, 5, 1000)
        print("45Left")

    def action90Left():
        rob.move(-10, 10, 1000)
        print("90Left")

    def actionBackwards():
        rob.move(-10, -10, 1000)
        print("Backwards")


    images = []

    for i in range(100):

        image = rob.get_image_front()
        images.append(image)
        for i, image in enumerate(images):
            cv2.imwrite('./src/week4/images/HW_dataset/img_p12-' + str(i) + ".png", image)

        # # Scaling IR signal
        # x = np.array((rob.read_irs()), dtype=float)
        # if np.amax(x, axis=0) != 0:
        #     x = x / np.amax(x, axis=0)
        #
        # NN = Neural_Network()
        # o = NN.forward(x)
        #
        # # Neural_Network()
        # NN.forward(x)
        #
        #
        # if 0 <= o < 0.1667:
        #     actionStraightForward()
        #     # time.sleep(0.1)
        #
        # elif 0.1667 <= o < 0.3334:
        #     action45Right()
        #     # time.sleep(0.1)
        #
        # elif 0.3334 <= o < 0.5:
        #     action90Right()
        #     # time.sleep(0.1)
        #
        # elif 0.5 <= o < 0.6668:
        #     action45Left()
        #     # time.sleep(0.1)
        #
        # elif 0.6668 <= o <= 0.8335:
        #     action90Left()
        #     # time.sleep(0.1)
        #
        # elif 0.8335 <= o <= 1:
        #     actionBackwards()
            # time.sleep(0.1)

        # print("Current IRS: \n" + str(rob.read_irs()))
        # print("Predicted Output: \n" + str(o))

    # pause the simulation and read the collected food
    # rob.pause_simulation()
    # print("Robobo collected {} food".format(rob.collected_food()))

    # stop the prey
    # prey_controller.stop()
    # prey_controller.join()

    # Stopping the simulation resets the environment
    # rob.stop_world()
