#!/usr/bin/env python2
from __future__ import print_function
import robobo
import cv2
import sys
import signal
import time
import numpy as np


if __name__ == "__main__":

    rob = robobo.SimulationRobobo().connect(address='192.168.1.103', port=19997)
    # rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.16")

    rob.play_simulation()

    # rob.set_emotion('happy')

    time.sleep(0.1)

    # for i in range(500):
    #     print(rob.read_irs())
    #     time.sleep(0.1)

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
            # self.hiddenSize = 3 # if using the other training weights
            # self.hiddenSize = 4 # if using third training weights
            self.hiddenSize = 5

            # weight matrix from input to hidden layer
            self.W1 = [

                # 100000 runs, 36 data points, 5 hidden notes
                # [1.9651708931413383, - 3.1468917450583964, 0.46486265139110255, 8.96958966049233, - 0.03139655851355883],
                # [- 3.1340990050987227, - 1.131108666308181, - 10.638442940981694, - 4.534165316178076, - 9.194693914334852],
                # [- 4.130189884648114, - 0.7497591160550472, 7.016781882584956, 1.6364095383796224, 6.344294277949948],
                # [- 0.8183015001562944, 0.4007759843227678, - 12.476092733420456, - 4.122802962507083, - 9.663554466055983],
                # [10.798169708461167, 0.3202361735191632, - 6.812743604633051, 0.6407613596471952, - 4.443586492171037],
                # [- 3.2138664670267514, - 12.46299725116111, 10.80581523859792, - 1.006677248078563, 3.179201146376463],
                # [- 6.384833310735183, - 2.8706120173664296, - 8.715828025978913, 11.290996439318432, - 3.7490884192425753],
                # [3.0280443539964796, 2.7345540898268843, - 15.28435966221402, - 7.812960932043841, 1.9268306564175899]

                # 10000 runs, 36 data points, 5 hidden nodes
                [-3.0293387728231327, - 6.6555223599347055, 1.3895854668586614, 0.6855186826623935, - 8.736955749205126],
                [- 0.9476957113266768, 7.668494276419445, - 0.7756409281619986, - 5.85312485132007, 0.25389602049900184],
                [0.8340066365985902, 4.659296270881088, 1.1990873969585283, - 2.677505497806095, - 0.15468326910182723],
                [- 1.598148715339809, - 2.5563229287603777, 1.4992281416978608, 1.348813447726324, - 1.500285219083036],
                [- 1.988141278050739, 2.6773660474162697, - 2.8807050972764805, - 2.6505088031395276, 4.745598997810229],
                [0.7819946716042375, 0.7308026519985253, 0.6452142776557535, - 1.0138534210771606, 4.047959517477389],
                [- 2.4681249872590607, 7.657084404350637, - 5.469860669901857, - 1.96862747401349, 4.418101806763281],
                [- 3.4538099829837674, - 5.896433777796302, - 2.141893493879237, 0.7941159461902955, 4.2268310061955905]

                # 1000 runs, 36 data points, 5 hidden nodes
                # [-2.140794969859636, - 1.6710521543136914, - 0.5733863603861561, - 1.4232311795814783, 0.7147924028092597],
                # [2.651884416401362, 0.1738993210378605, 0.37789697724159826, - 0.6723589593729654, - 0.8639383142188479],
                # [- 2.0574609812796947, - 0.7223212100686837, - 2.199157701379627, 1.8532420846850561, - 0.2960201314998339],
                # [- 6.658159851573543, - 4.657289297265272, 0.054408211469778076, - 0.027165806634827296, 0.5237649026617397],
                # [- 2.1478139695817515, - 0.16964958697508695, - 1.735154354076962, 0.8723606153118982, - 6.749167590616847],
                # [- 2.109432372714019, - 0.266404424977824, - 0.6796713892600329, 1.129288562065611, - 3.1571006778690323],
                # [1.1240321828722168, - 1.2197998468497206, - 1.1259105397172018, - 0.6558653498077667, - 1.8537733595260084],
                # [- 3.3072617640327393, - 3.184716575228105, - 0.6981271004371009, - 0.7324222286167046, 0.4600309929204798]

                # Fifth training: 1000 runs, 36 data points, 3 hidden nodes
                # [3.0188073985186112, 2.29095495940026, - 4.705923192820896],
                # [- 2.8497397568781033, - 0.12019012330642229, 3.7740975648694945],
                # [6.1430331836379555, - 0.48265693610843, - 3.923968142676809],
                # [- 0.14888335773998684, 4.1763224599401765, - 9.176948340807472],
                # [- 3.1946053843848783, 5.545815635674619, 2.480530680240116],
                # [- 3.2646382282236375, 3.817450873844489, - 0.7260128532130432],
                # [- 2.2352789797127235, 1.6302176268922495, 3.1220038354306663],
                # [0.7363767770316048, 4.557313374666476, - 1.5404328755192411]

                # Fourth training: 32 data point, 3 hidden nodes, if only back IRS then go straight forward
                # [-3.9486751361825765, 0.5393395516351593, 0.36344201023948564],
                # [11.134421827526468, - 0.468093553158263, - 1.343840326754022],
                # [30.52334103438082, 1.1771869917323694, 0.28798314613064996],
                # [5.870259718325996, - 0.184613663307973, - 0.07423230386169756],
                # [9.10835669997152, 0.07104238582528218, - 8.326058329172852],
                # [- 14.41135525467013, - 0.9857877997029163, - 12.245855391198004],
                # [18.892114455169, - 1.2027421132519602, - 41.40333760197874],
                # [- 16.965987086927417, 0.2351334469034928, 1.8799915366501059]

                # Third training with 32 data points and 4 hidden nodes instead of 3
                # [-20.270018337560153, 25.02915097655979, - 8.46106682047964, - 0.39167866358844017],
                # [- 12.842163683198033, - 4.332886109013558, - 4.911804289979967, - 1.2414744181368735],
                # [1.485223417532966, 0.7142245866633085, 4.125764920644568, 5.107484021941503],
                # [- 3.7392069832783004, - 3.4072136950401006, - 6.739586496570586, - 6.436072310502881],
                # [- 0.704671025190617, - 23.336688969612304, 2.7075388133278486, - 30.621176232410996],
                # [- 7.750935240583832, 3.209972373477899, - 15.70103199853893, - 2.5860391036045915],
                # [- 1.1586815767813268, - 22.242254668659616, - 4.0039236480518925, - 5.93664819110827],
                # [4.705236972103284, 32.59732515238087, 10.35600768433184, 10.971992252619037]

                # Second training with 27 data points, 3 hidden nodes
                # [-0.002146759633726726, 1.5787590090924117, 0.02755093914153383],
                # [2.064956323774703, 1.0106569692978253, - 0.7454973697182782],
                # [- 4.378883420583497, - 0.9557749101689611, 0.2539548827351401],
                # [0.6659387312932329, - 0.1956655485237519, 2.718272943344903],
                # [- 4.9122648867995675, 1.5150580881051798, 0.3741882228058275],
                # [- 5.0865488882269405, 0.34141380784035874, 0.40122846607067963],
                # [- 1.8822993170584819, 3.7415619420963515, - 0.7585516441173342],
                # [1.3577152071388003, 3.042204340075742, 4.31602724144434]

                # First training with 24 data points, 3 hiddes nodes
                # [3.244474565667816, - 0.012681924915328463, 7.829114223366257],
                # [0.6736972843037947, 6.480696649306626, 1.221171282458124],
                # [- 3.4376034753631255, 1.4647844406053094, 19.260754150576222],
                # [- 6.698380568379362, - 1.216035053805893, - 7.267912798590186],
                # [- 0.8861424550192054, 2.928440130924508, 1.8181142414469682],
                # [- 1.7347601422126557, 1.7857334506214941, 18.65554379685124],
                # [4.4409221615305725, 0.5928123637824768, 2.623152507233407],
                # [- 8.690382892353787, 0.2688325315047395, - 2.8978615601300866]
            ]

            # weight matrix from hidden to output layer
            self.W2 = [

                # 100000 runs, 36 data points, 5 hidden notes
                # [3.3841513594031922],
                # [- 7.324522392765753],
                # [- 18.820345993715026],
                # [- 2.5381205697176004],
                # [21.404828056553054]

                # 10000 runs, 36 data points, 5 hidden nodes
                [-2.219847820545832],
                [- 4.923262129595543],
                [3.6166335812156816],
                [- 4.381700423182979],
                [4.593004168524809]

                # 1000 runs, 36 data points, 5 hidden nodes
                # [-3.823658940297916],
                # [- 0.9627084365980388],
                # [5.879344947761717],
                # [- 0.3891004875083074],
                # [- 3.7585379384302016]

                # Fifth training: 36 data points, 3 hidden nodes
                # [-3.5204053412047607],
                # [2.3489590469335564],
                # [- 2.4961761949662837]

                # Fourth training: 32 data point, 3 hidden nodes, if only back IRS then go straight forward
                # [-2.487251632408659],
                # [4.939839595023239],
                # [- 5.36935266942341]

                # Third training with 32 data points and 4 hidden nodes instead of 3
                # [-20.566708816007466],
                # [2.285656897591394],
                # [21.33496409989751],
                # [- 5.981201945284517]

                # Second training with 27 data points, 3 hidden nodes
                # [-8.942729952717489],
                # [9.577495960037782],
                # [- 4.803584563434497]

                # First training with 24 data points, 3 hidden nodes
                # [-3.753038056513388],
                # [- 6.485502407222473],
                # [6.5890029347550625]
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


    for i in range(200):

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
