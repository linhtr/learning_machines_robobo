#!/usr/bin/env python2
from __future__ import print_function
import robobo
import time
import numpy as np

if __name__ == "__main__":

    rob = robobo.SimulationRobobo().connect(address='192.168.1.103', port=19997)
    #rob = robobo.HardwareRobobo(camera=False).connect(address="192.168.1.16")

    rob.play_simulation()

    #rob.set_emotion('happy')

    time.sleep(0.1)

    max_sensor = 1000

    # Scaling IR signal
    x = np.array((rob.read_irs()), dtype=float)
    print("input x:" + str(x))
    print("Begin IRS:" + str(rob.read_irs()))
    if np.amax(x, axis=0) != 0:
        x = (x / max_sensor)
        x = (np.log(x))
        print("log(IRS):" + str(x))

    class Neural_Network(object):
        def __init__(self):
            # parameters
            self.inputSize = 8
            self.outputSize = 1
            # self.hiddenSize = 3
            self.hiddenSize = 5

            # weight matrix from input to hidden layer
            self.W1 = [

                # Third: 1000 runs, 5 hidden nodes
                [1.2010150390970489, - 0.03161556199649709, - 0.2498545192841699, - 0.4133747157578791, - 0.18943443494823967],
                [- 1.109877066228962, 1.0425448425089932, - 0.8144251436047574, - 0.4153214962905958, 0.49183853467370814],
                [- 0.891195905714498, 2.7811434652797584, - 1.6034895364162383, - 2.192848127400721, - 2.2977826399130823],
                [- 0.20555456552152362, - 2.3092788991380053, 0.19533008912874777, - 0.9605383682603732, - 0.520100982086063],
                [- 2.864172742791945, - 1.8899719889511055, 0.26251415091201963, 1.043840310008703, 0.3789150464483622],
                [- 0.9347429692795381, - 0.39762840243997954, 0.6184027979416558, - 0.9050698907822967, - 0.11208839843868916],
                [0.9071356712914006, - 1.5341077005252517, 0.04722964441223146, 0.8679957383056037, 0.8795688513823339],
                [- 1.048581767734401, 0.40352636080009796, 2.2292155858063096, 0.6563587819517934, - 1.2777136440493442]

                # Second: 100000 runs, 5 hidden nodes
                # [-3.1985935953395557, - 3.4116959673766702, - 3.5673620170492937, - 8.633445712198073, - 2.1616118502362207],
                # [- 0.1311233339374687, - 5.342413050253929, 4.862715738041956, 9.773426978275754, 0.8287710855708983],
                # [- 4.5499632387979165, 37.887406595320556, 6.316342950225569, - 25.281782578848645, - 11.711044921839097],
                # [10.512307424546805, - 33.186789290120316, 8.39227169406332, - 5.7744368204498775, 4.823694175628503],
                # [5.387050769186433, - 17.393572219343536, 4.295316182012715, 9.70956247200192, 0.2805495796632925],
                # [2.2012176578045466, 0.5878711693417609, 4.864871935736074, - 0.36135089929490355, - 8.318689365732247],
                # [1.4620934185729124, - 3.1032636330161023, 0.6008034673983084, 3.5141385695271854, 3.3945482169373316],
                # [6.185589432421932, - 1.111294867684408, 0.08615341112437515, 15.843581027791046, 3.928981203943643]

                # First: 100000 runs, 3 hidden nodes
                # [10.10882821677443,17.052784199771413,10.558992083337108],
                # [-2.004274963522354,16.36305513126182,1.4844978445760182],
                # [10.560231354107746,37.84546047807807,14.455612487815378],
                # [26.79594066360207,-13.260715702941006,5.398567972100232],
                # [-14.56194057330128,-20.903228424343265,-10.270435830586708],
                # [-6.084246748513748,-6.298312517756375,8.173829609505907],
                # [-3.706029526023833,5.382397299716315,2.728576033430334],
                # [-30.04656435036936,16.131151293033426,-7.644457013337049]

            ]

            # weight matrix from hidden to output layer
            self.W2 = [
                # 1000 runs, 5 hidden nodes
                [-1.927712124059419],
                [- 2.7143424285922384],
                [2.0483119103055247],
                [1.2106048620352547],
                [1.0441099295067815]

                # 100000 runs, 5 hidden nodes
                # [10.113870055949647],
                # [- 10.344987920829356],
                # [8.66035586757982],
                # [- 19.872470145875663],
                # [8.502108836085181]

                # [-7.697670719707903],
                # [-8.454400245394526],
                # [14.310545551680866]
            ]

        def forward(self, x):
            # forward propagation through our network
            self.z = np.dot(x, self.W1)  # dot product of x (input) and first set of weights
            self.z2 = self.sigmoid(self.z)  # activation function
            self.z3 = np.dot(self.z2, self.W2)  # dot product of hidden layer (z2) and second set of weights
            o = self.sigmoid(self.z3)  # final activation function
            return o

        def sigmoid(self, s):
            # activation function
            return 1 / (1 + np.exp(-s))


    for i in range(1000):

        # Scaling IR signal
        x = np.array((rob.read_irs()), dtype=float)
        if np.amax(x, axis=0) != 0:
            x = x / max_sensor
            x = np.log(x)

        NN = Neural_Network()
        o = NN.forward(x)

        Neural_Network()

        if 0 <= o < 0.1667:
            rob.move(20, 20, 1000)
            print("Straight")
            time.sleep(0.1)

        elif 0.1667 <= o < 0.3334:
            rob.move(20, -5, 1000)
            print("45right")
            time.sleep(0.1)

        elif 0.3334 <= o < 0.5:
            rob.move(50, -5, 1000)
            print("90right")
            time.sleep(0.1)

        elif 0.5 <= o < 0.6668:
            rob.move(-5, 20, 1000)
            print("45left")
            time.sleep(0.1)

        elif 0.6668 <= o <= 0.8335:
            rob.move(-5, 50, 1000)
            print("90left")
            time.sleep(0.1)

        elif 0.8335 <= o <= 1:
            rob.move(-20, -20, 1000)
            print("Backwards")
            time.sleep(0.1)

        print("Current IRS: \n" + str(rob.read_irs()))
        print("Predicted output: \n" + str(o))
