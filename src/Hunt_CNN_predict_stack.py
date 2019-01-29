#!/usr/bin/env python2
from __future__ import print_function
import robobo
import cv2
import sys
import signal
import time
import os

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
from collections import deque
import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    loaded_model = load_model('./src/week4/models/CNN_Sim_weights02-0.62.hdf5')
    rob = robobo.SimulationRobobo().connect(address='192.168.1.13.', port=19999)
    rob.play_simulation()

    # connect to prey robot
    prey_robot = robobo.SimulationRoboboPrey().connect(address='192.168.1.13.', port=19989)
    # initialise class prey
    prey_controller = prey.Prey(robot=prey_robot)
    # start the thread prey, makes the prey move
    prey_controller.start()

    rob.set_phone_tilt(32, 100)
    time.sleep(0.1)

    def actionStraightForward():
        rob.move(20, 20, 400)
        print("StraightForward")

    def action45Right():
        rob.move(5, -5, 400)
        print("45Right")

    def action90Right():
        rob.move(10, -10, 400)
        print("90Right")

    def action45Left():
        rob.move(-5, 5, 400)
        print("45Left")

    def action90Left():
        rob.move(-10, 10, 400)
        print("90Left")

    def actionBackwards():
        rob.move(-10, -10, 200)
        print("Backwards")

    # images = []

    for i in range(100):

        predict_image = rob.get_image_front()
        # print(predict_image)

        # Use this to save images for database
        # images.append(predict_image)
        # for i, predict_image in enumerate(images):
        #     cv2.imwrite('./src/week3/images/run/img-' + str(i) + ".png", predict_image)

        # temporarily save image to computer and load image
        cv2.imwrite('./src/week3/images/run/img-0.png', predict_image)
        predict_image = cv2.imread('./src/week3/images/run/img-0.png')
        # print(predict_image)
        predict_image = cv2.resize(predict_image, (64, 64))
        predict_image = predict_image[..., ::-1].astype(np.float32) / 255.0
        predict_image = image.img_to_array(predict_image)
        predict_image = np.expand_dims(predict_image, axis=0)  # Add fourth dimension

        # predicted output
        output = loaded_model.predict_classes(predict_image)
        print("Predicted output:" + str(output))

        if output[0] == 0:
            prediction = 'straight'
            actionStraightForward()
        elif output[0] == 1:
            prediction = '45right'
            action45Right()
        elif output[0] == 2:
            prediction = '90right'
            action90Right()
        elif output[0] == 3:
            prediction = '45left'
            action45Left()
        elif output[0] == 4:
            prediction = '90left'
            action90Left()
        elif output[0] == 5:
            prediction = 'back'
            actionBackwards()

        # Remove temporarily image from computer
        try:
            os.remove('./src/week3/images/run/img-0.png')
        except:
            pass


    # Stopping the simulation resets the environment
    rob.stop_world()


#####################################################################################################################

def reward(red_percentage):
    if red_percentage >= 0.10:
        return 10
    elif 0.05 >= red_percentage > 0.10:
        return 5
    else:
        return 0

stack_size = 4  # We stack 4 frames

# Initialize deque with zero-images one array for each image
stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)


def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame, red_percentage = preprocess_frame(state)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames


class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)

        return [self.buffer[i] for i in index]


# Instantiate memory
memory = Memory(max_size=memory_size)

memory.add((state, action, reward, next_state, done))

### LEARNING PART
# Obtain random mini-batch from memory
batch = memory.sample(batch_size)
states_mb = np.array([each[0] for each in batch], ndmin=3)
actions_mb = np.array([each[1] for each in batch])
rewards_mb = np.array([each[2] for each in batch])
next_states_mb = np.array([each[3] for each in batch], ndmin=3)
dones_mb = np.array([each[4] for each in batch])

target_Qs_batch = []

# Get Q values for next_state
Qs_next_state = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: next_states_mb})

# Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
for i in range(0, len(batch)):
    terminal = dones_mb[i]

    # If we are in a terminal state, only equals reward
    if terminal:
        target_Qs_batch.append(rewards_mb[i])

    else:
        target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
        target_Qs_batch.append(target)

targets_mb = np.array([each for each in target_Qs_batch])

loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                   feed_dict={DQNetwork.inputs_: states_mb,
                              DQNetwork.target_Q: targets_mb,
                              DQNetwork.actions_: actions_mb})

# Write TF Summaries
summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                        DQNetwork.target_Q: targets_mb,
                                        DQNetwork.actions_: actions_mb})
writer.add_summary(summary, episode)
writer.flush()

# Save model every 5 episodes
if episode % 5 == 0:
    save_path = saver.save(sess, "./week4/models/model.ckpt")
    print("Model Saved")

    saver.restore(sess, "./week4/models/model.ckpt")