#!/usr/bin/env python2
import glob
from PIL import Image
from keras.preprocessing import image
import cv2

# images = glob.glob("week4/images/predict/*.png")
# for image in images:
#     with open(image, 'rb') as file:
#         img = Image.open(file)
#         area = (0, 100, 480, 640)
#         cropped_img = img.crop(area)
#         print(cropped_img.size)
#         # cropped_img.show()
#         # cropped_img.save(img + str(image) + ".png")
#         cropped_img.save(image)


# test_image = image.load_img('week4/images/predict/img_p13-43.png', target_size = (64, 64))
# test_image.show()
# test_image.save('week4/images/predict/test' + '.png')

# img = Image.open('week4/images/predict/img_p13-43.png')
# area = (0, 60, 480, 540)
# cropped_img = img.crop(area)
# print(cropped_img.size)
# # cropped_img.show()
# cropped_img.save('week4/images/predict/test2' + ".png")

# test_image = image.load_img('week4/images/predict/img_p13-43.png', target_size = (48, 54))
# test_image.show()
# test_image.save('week4/images/predict/test3' + '.png')

# test_image = image.load_img('week4/images/predict/img_p13-43.png', target_size = (54, 48))
# test_image.show()
# test_image.save('week4/images/predict/test4' + '.png')

# img = Image.open('week4/images/predict/img_p6-13.png')
# area = (0, 100, 480, 640)
# cropped_img = img.crop(area)
# print(cropped_img.size)
# # cropped_img.show()
# cropped_img.save('week4/images/predict/test5' + ".png")
#
# test_image = image.load_img('week4/images/predict/test5.png', target_size = (64, 64))
# test_image.show()
# test_image.save('week4/images/predict/test6' + '.png')

# img = Image.open('week4/images/predict/img_p9-14.png')
# area = (0, 100, 480, 640)
# cropped_img = img.crop(area)
# print(cropped_img.size)
# # cropped_img.show()
# cropped_img.save('week4/images/predict/test9' + ".png")
#
# test_image = image.load_img('week4/images/predict/test9.png', target_size = (64, 64))
# # test_image.show()
# test_image.save('week4/images/predict/test10' + '.png')

predict_image = cv2.imread('week4/images/predict/test9.png')
predict_image = cv2.resize(predict_image, (64, 64))
cv2.imshow('image', predict_image)
cv2.imwrite('week4/images/predict/test11.png', predict_image)

# for i in range(6):
#     i = glob.glob("week4/images/HW_dataset_copy/training_set/" + str(i) + "/*.png")
#     for image in i:
#         with open(image, 'rb') as file:
#             img = Image.open(file)
#             area = (0, 100, 480, 640)
#             cropped_img = img.crop(area)
#             print(cropped_img.size)
#             cropped_img.save(image)
