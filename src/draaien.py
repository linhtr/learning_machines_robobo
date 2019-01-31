#!/usr/bin/env python2
from __future__ import print_function
import robobo
import time
import numpy as np

if __name__ == "__main__":
    rob = robobo.HardwareRobobo(camera=False).connect(address="172.20.10.11")

#rechtdoor
#rob.move(20, 20, 1000)
#time.sleep(0.1)

# 30 graden naar rechts
rob.move(15, -5, 1000)

# 30 graden naar links
rob.move(-5, 15, 1000)

# 120 graden naar links
rob.move(-5, 90, 1000)

# 90 graden naar rechts
rob.move(55, -5, 1000)

# 20 graden naar rechts 
# rob.move(10, -5, 1000)

# 45 graden naar rechts
# rob.move(20, -5, 1000)

# 45 graden naar links
#rob.move(-5, 20, 1000)

# 90 graden naar links
#rob.move(-5, 50, 1000)

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



