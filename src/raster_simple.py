import cv2
import numpy as np

image = cv2.imread('sunflower.jpg')
cv2.imshow('Original Image', image)
cv2.waitKey(0) 

height, width = image.shape[:2] # save height and width once

## store all coordinates
start_row_tl, start_col_tl = int(0), int(0)
end_row_tl, end_col_tl = int(height * .5), int(width * .5)
start_row_tr, start_col_tr = int(0), int(width * .5)
end_row_tr, end_col_tr = int(height * .5), int(width)
start_row_bl, start_col_bl = int(height *.5), int(0)
end_row_bl, end_col_bl = int(height), int(width * .5)
start_row_br, start_col_br = int(height *.5), int(width * .5)
end_row_br, end_col_br = int(height), int(width)


## store 4 images
cropped_top_left = image[start_row_tl:end_row_tl , start_col_tl:end_col_tl]
cropped_top_right = image[start_row_tr:end_row_tr , start_col_tr:end_col_tr]
cropped_bot_left = image[start_row_bl:end_row_bl , start_col_bl:end_col_bl]
cropped_bot_right = image[start_row_br:end_row_br , start_col_br:end_col_br]

