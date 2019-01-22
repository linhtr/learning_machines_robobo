#!/usr/bin/env python2

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential, model_from_json, load_model
from keras.layers import Convolution2D as Conv2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

from IPython.display import display
from PIL import Image

import numpy as np
from keras.preprocessing import image

# Initialize the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 6, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the images
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'images/dataset/training_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'categorical'
)

test_set = test_datagen.flow_from_directory(
    'images/dataset/test_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'categorical'
)

classifier.fit_generator(
    training_set,
    steps_per_epoch = 681, #number of training images
    epochs = 10,
    validation_data = test_set,
    validation_steps = 800
)

# serialize model to JSON
#  the keras model which is trained is defined as 'model' in this example
model_json = classifier.to_json()


with open("CNN_weights.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
classifier.save_weights("CNN_weights.h5")


# Test a random image
# Go 45 right
test_image = image.load_img('images/p2_fail/img_p2-190.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
# Go straight
test_image2 = image.load_img('images/p2_fail/img_p2-1.png', target_size = (64, 64))
test_image2 = image.img_to_array(test_image2)
test_image2 = np.expand_dims(test_image2, axis = 0)
# Go backward
test_image3 = image.load_img('images/p2_fail/img_p2-36.png', target_size = (64, 64))
test_image3 = image.img_to_array(test_image3)
test_image3 = np.expand_dims(test_image3, axis = 0)

result = classifier.predict(test_image)
result2 = classifier.predict(test_image2)
result3 = classifier.predict(test_image3)

print(result)
print(result2)
print(result3)

training_set.class_indices
if result[0][0] == 1:
    prediction = 'straight'
elif result[0][1] == 1 :
    prediction = '45right'
elif result[0][2] == 1:
    prediction = '90right'
elif result[0][3] == 1:
    prediction = '45left'
elif result[0][4] == 1:
    prediction = '90left'
elif result[0][5] == 1:
    prediction = 'back'

if result2[0][0] == 1:
    prediction2 = 'straight'
elif result2[0][1] == 1 :
    prediction2 = '45right'
elif result2[0][2] == 1:
    prediction2 = '90right'
elif result2[0][3] == 1:
    prediction2 = '45left'
elif result2[0][4] == 1:
    prediction2 = '90left'
elif result2[0][5] == 1:
    prediction2 = 'back'

if result3[0][0] == 1:
    prediction3 = 'straight'
elif result3[0][1] == 1 :
    prediction3 = '45right'
elif result3[0][2] == 1:
    prediction3 = '90right'
elif result3[0][3] == 1:
    prediction3 = '45left'
elif result3[0][4] == 1:
    prediction3 = '90left'
elif result3[0][5] == 1:
    prediction3 = 'back'

print(prediction)
print(prediction2)
print(prediction3)