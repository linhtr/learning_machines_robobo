#!/usr/bin/env python2

import numpy as np


# x = [BR, BC, BL, FRR, FR, FC, FL, FLL], y = action
x = np.array (([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [11.0, 16.0, 7.0, 69.0, 122.0, 187.0, 125.0, 21.0],
        [8.0, 19.0, 31.0, 88.0, 198.0, 334.0, 215.0, 120.0],
        [7.0, 18.0, 26.0, 86.0, 206.0, 317.0, 201.0, 120.0],
        [8.0, 17.0, 7.0, 9.0, 43.0, 25.0, 38.0, 8.0],
        [18.0, 81.0, 35.0, 9.0, 41.0, 8.0, 35.0, 7.0],
        [23.0, 80.0, 26.0, 27.0, 47.0, 8.0, 35.0, 8.0],
        [24.0, 83.0, 26.0, 8.0, 41.0, 8.0, 35.0, 7.0],
        [7.0, 17.0, 7.0, 15.0, 52.0, 16.0, 36.0, 8.0],
        [8.0, 17.0, 7.0, 8.0, 42.0, 14.0, 47.0, 10.0],
        [7.0, 17.0, 7.0, 14.0, 199.0, 5125.0, 1348.0, 204.0],
        [7.0, 18.0, 7.0, 9.0, 41.0, 23.0, 38.0, 16.0],
        [30.0, 257.0, 50.0, 8.0, 41.0, 8.0, 35.0, 8.0],
        [28.0, 22.0, 8.0, 8.0, 44.0, 11.0, 36.0, 7.0],
        [19.0, 20.0, 8.0, 8.0, 41.0, 9.0, 47.0, 14.0],
        [19.0, 19.0, 8.0, 26.0, 54.0, 9.0, 36.0, 8.0],
        [8.0, 18.0, 19.0, 8.0, 41.0, 8.0, 40.0, 15.0],
        [8.0, 17.0, 8.0, 9.0, 40.0, 8.0, 36.0, 8.0],
        [23.0, 18.0, 23.0, 28.0, 46.0, 7.0, 37.0, 31.0],
        [13.0, 19.0, 25.0, 19.0, 42.0, 8.0, 38.0, 28.0],
        [5.0, 12.0, 5.0, 6.0, 27.0, 8.0, 42.0, 13.0],
        [5.0, 12.0, 13.0, 6.0, 27.0, 8.0, 53.0, 32.0],
        [5.0, 12.0, 5.0, 16.0, 40.0, 7.0, 25.0, 5.0],
        [26.0, 11.0, 5.0, 29.0, 45.0, 7.0, 25.0, 5.0],
        [5.0, 11.0, 5.0, 21.0, 41.0, 7.0, 35.0, 21.0],
        [7.0, 59.0, 22.0, 7.0, 26.0, 6.0, 33.0, 14.0],
        [29.0, 67.0, 31.0, 6.0, 25.0, 5.0, 22.0, 5.0],
        [17.0, 42.0, 28.0, 15.0, 25.0, 5.0, 31.0, 28.0],
        [69.0, 43.0, 10.0, 6.0, 27.0, 6.0, 22.0, 5.0],
        [67.0, 38.0, 11.0, 6.0, 26.0, 5.0, 22.0, 5.0],
        [5.0, 22.0, 114.0, 5.0, 27.0, 7.0, 23.0, 4.0],
        [8.0, 14.0, 25.0, 5.0, 24.0, 5.0, 23.0, 5.0],
        [17.0, 17.0, 35.0, 59.0, 57.0, 9.0, 38.0, 26.0],
        [17.0, 17.0, 10.0, 62.0, 58.0, 9.0, 37.0, 10.0],
        [8.0, 17.0, 27.0, 12.0, 40.0, 9.0, 42.0, 51.0],
        [13.0, 18.0, 7.0, 1073.0, 7895.0, 28623.0, 8028.0, 651.0],
        [13.0, 17.0, 8.0, 131.0, 774.0, 22422.0, 27115.0, 4332.0],
        [12.0, 17.0, 7.0, 5615.0, 35505.0, 27856.0, 1295.0, 93.0],
        [10.0, 18.0, 16.0, 9.0, 42.0, 9.0, 40.0, 10.0]),dtype=float)

y = np.array(([1], [5], [3], [3], [3], [1], [1], [1], [4], [2], [2], [2], [1], [1], [2], [5], [3], [1], [1], [1], [2], [3], [4], [5], [3], [3], [1], [1], [1], [1], [1], [1], [1], [1], [1], [6], [6], [6], [1]), dtype=float)

# Go backwards
xPredicted = np.array(([11.0, 33.0, 8.0, 2642.0, 593.0, 35.0, 31.0, 8.0]), dtype=float) # our input data for the prediction

# scale units

x = x/np.amax(x, axis=0)
xPredicted = xPredicted/np.amax(xPredicted, axis=0) # maximum of xPredicted (our input data for the prediction)

# max_sensor = 1000
# if np.amax(x, axis=0) != 0:
#     x = x / max_sensor
#     x = np.log(x)
# if np.amax(xPredicted, axis=0) != 0:
#     xPredicted = xPredicted / max_sensor
#     xPredicted = np.log(xPredicted) # scaling xPredicted

y = y/7

class Neural_Network(object):
    def __init__(self):
        # parameters
        self.inputSize = 8
        self.outputSize = 1
        # self.hiddenSize = 3
        self.hiddenSize = 5

        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (8x3) or (8x5) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) or (5x1) weight matrix from hidden to output layer

    def forward(self, x):
        # forward propagation through our network
        self.z = np.dot(x, self.W1)  # dot product of x (input) and first set of 8x3 weights
        self.z2 = self.sigmoid(self.z)  # activation function
        self.z3 = np.dot(self.z2, self.W2)  # dot product of hidden layer (z2) and second set of 3x1 weights
        o = self.sigmoid(self.z3)  # final activation function
        return o

    def sigmoid(self, s):
        # activation function
        return 1 / (1 + np.exp(-s))

    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)

    def backward(self, x, y, o):
        # backward propagate through the network
        self.o_error = y - o  # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o)  # applying derivative of sigmoid to error

        self.z2_error = self.o_delta.dot(self.W2.T)  # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)  # applying derivative of sigmoid to z2 error

        self.W1 += x.T.dot(self.z2_delta)  # adjusting first set (input --> hidden) weights
        self.W2 += self.z2.T.dot(self.o_delta)  # adjusting second set (hidden --> output) weights

    def train(self, x, y):
        o = self.forward(x)
        self.backward(x, y, o)

    def saveWeights(self):
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")

    def predict(self):
        print ("Predicted data based on trained weights: ");
        print ("Input (scaled): \n" + str(xPredicted));
        print ("Output: \n" + str(self.forward(xPredicted)));

NN = Neural_Network()
for i in range(1000): # trains the NN 1,000 times
    print ("# " + str(i) + "\n")
    print ("Input: \n" + str(x))
    print ("Actual Output: \n" + str(y))
    print ("Predicted Output: \n" + str(NN.forward(x)))
    print ("Loss: \n" + str(np.mean(np.square(y - NN.forward(x))))) # mean sum squared loss
    print ("\n")
    NN.train(x, y)

NN.saveWeights()
NN.predict()
