#!/usr/bin/env python2

import numpy as np


# x = [BR, BC, BL, FRR, FR, FC, FL, FLL], y = action
x = np.array ((
        [0, 0, 0, 0, 0, 0.18, 0, 0],
        [0.17, 0, 0, 0.13, 0.10, 0.09, 0.10, 0.13],
        [0, 0.20, 0.14, 0.10, 0.13, 0, 0, 0],
        [0, 0, 0.16, 0.17, 0.13, 0.11, 0.13, 0.16],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0.12, 0, 0, 0.18, 0, 0, 0, 0.19],
        [0, 0, 0.17, 0, 0, 0, 0, 0],
        [0.09, 0.06, 0.09, 0, 0, 0.20, 0, 0],
        [0.16, 0, 0, 0, 0, 0, 0.17, 0.11],
        [0, 0, 0, 0, 0, 0, 0.11, 0.08],
        [0.16, 0, 0, 0.12, 0.19, 0, 0, 0],
        [0, 0, 0.09, 0, 0, 0.10, 0.11, 0],
        [0, 0.12, 0, 0, 0.14, 0.18, 0, 0],
        [0.13, 0, 0, 0.12, 0.10, 0.09, 0, 0],
        [0.06, 0.04, 0.06, 0, 0, 0, 0, 0],
        [0, 0, 0, 0.06, 0.04, 0.03, 0.04, 0.06],
        [0.05, 0.03, 0.05, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.20, 0.10, 0, 0],
        [0.11, 0.10, 0.07, 0, 0, 0, 0, 0],
        [0.16, 0, 0, 0.18, 0.18, 0, 0, 0],
        [0.10, 0, 0, 0, 0, 0, 0.14, 0.09],
        [0, 0, 0, 0.06, 0.06, 0.09, 0.10, 0.10],
        [0, 0.12, 0, 0, 0, 0.12, 0.08, 0.08],
        [0, 0, 0, 0, 0, 0, 0.12, 0.12],
        [0, 0, 0, 0.030998928743839738, 0.016018293278596683, 0.0075293421075112825, 0.015680089994701277, 0.030452554611877441],
        [0, 0, 0, 0.044369299601287764, 0.027294445529797661, 0.018116073949289074, 0.026956025372510378, 0.043821774080371245],
        [0, 0, 0, 0.045464526426287337, 0.025166344527765731, 0.013092007885887288, 0.018679134482676208, 0.031091834764664525],
        [0, 0, 0.17166810001442021, 0.16204426537027639, 0, 0, 0, 0],
        [0.12036189605490287, 0, 0, 0, 0.19467904414736378, 0, 0, 0],
        [0.069914638619214417, 0.092383508466895276, 0, 0, 0, 0, 0, 0],
        [0.068940018842938502, 0.080505634362549067, 0, 0.061257055010963797, 0.073187679002419792, 0.13286638015435187, 0, 0],
        [0, 0.17946625429399593, 0, 0, 0, 0, 0, 0],
        [0.16286454461721731, 0.18537926350864439, 0.17582401609935699, 0, 0, 0, 0, 0],
        [0, 0, 0, 0.054840390641454448, 0.036121749499455021, 0.026399334514012895, 0.035774900485986014, 0.054272044716290511],
        [0, 0, 0, 0.093592721296320597, 0.053889857747429645, 0.03051781011925744, 0.030697875937627912, 0.039066851495507815],
        [0, 0, 0, 0.056498322280424833, 0.044417532627750216, 0.041490685014241607, 0.06246542421845333, 0.099048974966916836]), dtype=float)

# Actions array
y = np.array(([5], [5], [4], [3], [1], [1], [1], [3], [2], [2], [4], [2], [3], [5], [1], [3], [1], [3], [1], [4], [3], [3], [3], [2], [5], [5], [3], [1], [4], [1], [5], [1], [1], [6], [6], [6]), dtype=float)

# 45 degrees to the right
# xPredicted = np.array(([0.12549120744041711, 0, 0, 0, 0, 0, 0.12782381847169746, 0.093425879694824576]), dtype=float)

# 90 degrees to the left
xPredicted = np.array(([0.11444473173043691, 0.077040054291506621, 0, 0.11989665728397389, 0.10076439963513562, 0, 0, 0]), dtype=float)

# scale units
x = x/np.amax(x, axis=0)
xPredicted = xPredicted/np.amax(xPredicted, axis=0) # maximum of xPredicted (our input data for the prediction)
y = y/7

class Neural_Network(object):
    def __init__(self):
        # parameters
        self.inputSize = 8
        self.outputSize = 1
        # self.hiddenSize = 3
        self.hiddenSize = 5


        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (8x5) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (5x1) weight matrix from hidden to output layer

    def forward(self, x):
        # forward propagation through our network
        self.z = np.dot(x, self.W1)  # dot product of x (input) and first set of 8x5 weights
        self.z2 = self.sigmoid(self.z)  # activation function
        self.z3 = np.dot(self.z2, self.W2)  # dot product of hidden layer (z2) and second set of 5x1 weights
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

        self.z2_error = self.o_delta.dot(
            self.W2.T)  # z2 error: how much our hidden layer weights contributed to output error
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
for i in range(100000): # trains the NN ... times
    if i == 1:
        print ("# " + str(i) + "\n")
        print ("Input: \n" + str(x))
        print ("Actual Output: \n" + str(y))
        print ("Predicted Output: \n" + str(NN.forward(x)))
        print ("Loss: \n" + str(np.mean(np.square(y - NN.forward(x))))) # mean sum squared loss
        print ("\n")
    if i == 99999:
        print ("# " + str(i) + "\n")
        print ("Input: \n" + str(x))
        print ("Actual Output: \n" + str(y))
        print ("Predicted Output: \n" + str(NN.forward(x)))
        print ("Loss: \n" + str(np.mean(np.square(y - NN.forward(x))))) # mean sum squared loss
        print ("\n")
    NN.train(x, y)

NN.saveWeights()
NN.predict()
