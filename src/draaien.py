#!/usr/bin/env python2
from __future__ import print_function
import robobo
import time
import numpy as np

if __name__ == "__main__":
    rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.21")

    # rechtdoor
    # rob.move(20, 20, 1000)
    # time.sleep(0.1)

    # 45 graden naar rechts
    # rob.move(20, -5, 1000)
    rob.move(10, -5, 1000)

    # 90 graden naar rechts
    # rob.move(50, -5, 1000)

    # 45 graden naar links
    # rob.move(-5, 20, 1000)
    # rob.move(-5, 10, 1000)

    # 90 graden naar links
    # rob.move(-5, 50, 1000)

        # if 0 <= o < 0.2:
        #     rob.move(20, 20, 1000)
        #
        # elif 0.2 <= o < 0.4:
        #     rob.move(10, -5, 1000)
        #
        # elif 0.4 <= o < 0.6:
        #     rob.move(10, -10, 1000)
        #
        # elif 0.6 <= o < 0.8:
        #     rob.move(-5, 10, 1000)
        #
        # elif 0.8 <= o <= 1:
        #     rob.move(-10, 10, 1000)
        #
        # time.sleep(0.1)



