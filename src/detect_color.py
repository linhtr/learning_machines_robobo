# import the necessary packages
import numpy as np
import argparse
import cv2
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())
 
# load the image
image = cv2.imread(args["image"])
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# create NumPy arrays from the boundaries
greenUpper = np.array([86,255,255], dtype = "uint8")
greenLower = np.array([36,0,0], dtype = "uint8")

# find the colors within the specified boundaries and apply
# the mask
#mask = cv2.inRange(image, lower, upper)
#mask = cv2.inRange(hsv, greenLower, greenLower)
mask = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))
#output = cv2.bitwise_and(image, image, mask = mask)
output = cv2.bitwise_and(hsv, hsv, mask = mask)


# show the images
cv2.imshow("images", np.hstack([image, output]))
cv2.waitKey(0)