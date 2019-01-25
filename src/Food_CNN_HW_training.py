#!/usr/bin/env python2

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential, model_from_json, load_model
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

from IPython.display import display
from PIL import Image

import numpy as np
from keras.preprocessing import image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Initialize the CNN
classifier = Sequential()

# Step 1 - Convolution
#number of filters (32), shape for each filter (3, 3), input shape (64, 64), type of image(RGB(3) or B/W), activation function
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = "relu"))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(activation = "relu", units = 128)) #output_dim = 128
classifier.add(Dense(activation = "softmax", units = 6)) #output_dim = 6

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the images
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'week3/images/HW_dataset/training_set',
    target_size = (64, 64),
    batch_size = 412, #Number of observations per batch
    class_mode = 'categorical'
)

test_set = test_datagen.flow_from_directory(
    'week3/images/HW_dataset/test_set',
    target_size = (64, 64),
    batch_size = 103, #Number of observations per batch
    class_mode = 'categorical'
)

# Checkpoint
filepath = "week3/models/CNN_HW_weights(11){epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
callbacks_list = [checkpoint]

# Train convolutional neural network
history = classifier.fit_generator(
    training_set,
    steps_per_epoch = 412, #Number of training images
    epochs = 2, #1 epoch means neural network is trained on every training examples in 1 pass --> training cycle
    validation_data = test_set,
    validation_steps = 1 #suggestion: validation_steps = TotalvalidationSamples / ValidationBatchSize (103:32) of 103:103
)

# Get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
fig_Loss = plt.figure()
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
fig_Loss.suptitle('Loss History', fontsize=18)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();
fig_Loss.savefig('week3/figures/fig_HW_LossHistory(11).png')

# Visualize accuracy history
fig_Acc = plt.figure()
plt.plot(epoch_count, training_acc, 'r--')
plt.plot(epoch_count, test_acc, 'b-')
plt.legend(['Training Accuracy', 'Test Accuracy'])
fig_Acc.suptitle('Accuracy History', fontsize=18)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show();
fig_Acc.savefig('week3/figures/fig_HW_AccHistory(11).png')

# serialize model to JSON
# the keras model which is trained is defined as 'model' in this example
model_json = classifier.to_json()
with open("week3/models/CNN_HW_model(11).json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("week3/models/CNN_HW_weights(11).h5")
print("Saved model to disk")

# Test a random image
# Go straight
test_image = image.load_img('week3/images/predict/HW_p1-3.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0) # Add fourth dimension
# Go 45 left
test_image2 = image.load_img('week3/images/predict/HW_p1-10.png', target_size = (64, 64))
test_image2 = image.img_to_array(test_image2)
test_image2 = np.expand_dims(test_image2, axis = 0)
# Go 45 right
test_image3 = image.load_img('week3/images/predict/HW_p5-55.png', target_size = (64, 64))
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
