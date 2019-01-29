#!/usr/bin/env python2
# TensorFlow and tf.keras
import tensorflow as tf
import robobo
import prey
import cv2
import random
import time

from tensorflow import keras
from keras.models import Sequential, model_from_json, load_model
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

from skimage import transform
from skimage.color import rgb2gray
from collections import deque

import numpy as np
from keras.preprocessing import image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

done = True

def connect_VREP():
    rob = robobo.SimulationRobobo().connect(address='130.37.247.69', port=19997)
    prey_robot = robobo.SimulationRoboboPrey().connect(address='130.37.247.69', port=19989)

def create_environment():

    connect_VREP()

    rob.play_simulation()
    rob.set_phone_tilt(32, 100)
    # connect to prey robot
    # initialise class prey
    prey_controller = prey.Prey(robot=prey_robot)
    # start the thread prey, makes the prey move
    prey_controller.start()


    straight = [1, 0, 0, 0, 0, 0, 0, 0]
    right20 = [0, 1, 0, 0, 0, 0, 0, 0]
    right45 = [0, 0, 1, 0, 0, 0, 0, 0]
    right90 = [0, 0, 0, 1, 0, 0, 0, 0]
    left20 = [0, 0, 0, 0, 1, 0, 0, 0]
    left45 = [0, 0, 0, 0, 0, 1, 0, 0]
    left90 = [0, 0, 0, 0, 0, 0, 1, 0]
    back = [0, 0, 0, 0, 0, 0, 0, 1]

    possible_actions = [straight, right20, right45, right90, left20, left45, left90, back]

    return rob, possible_actions

def test_environment():
    connect_VREP()
    # rob.play_simulation()
    # rob.set_phone_tilt(32, 100)
    # prey_controller = prey.Prey(robot=prey_robot)
    # prey_controller.start()

    straight = [1, 0, 0, 0, 0, 0, 0, 0]
    right20 = [0, 1, 0, 0, 0, 0, 0, 0]
    right45 = [0, 0, 1, 0, 0, 0, 0, 0]
    right90 = [0, 0, 0, 1, 0, 0, 0, 0]
    left20 = [0, 0, 0, 0, 1, 0, 0, 0]
    left45 = [0, 0, 0, 0, 0, 1, 0, 0]
    left90 = [0, 0, 0, 0, 0, 0, 1, 0]
    back = [0, 0, 0, 0, 0, 0, 0, 1]
    actions = [straight, right20, right45, right90, left20, left45, left90, back]

    episodes = 10
    for i in range(episodes):
        # rob.new_episode()
        rob.play_simulation()
        rob.set_phone_tilt(32, 100)
        prey_controller = prey.Prey(robot=prey_robot)
        prey_controller.start()
        # while not rob.is_episode_finished():
        state = rob.get_front_image()
        img = state.screen_buffer
        #misc = state.game_variables
        action = random.choice(actions)
        print(action)
        # reward = rob.move(action)
        reward = rob_action(action)
        print ("\treward:", reward)
        time.sleep(0.02)
        print ("Result:", rob.get_total_reward())
        time.sleep(2)
    rob.stop_world()

rob, possible_actions = create_environment()


def preprocess_frame(frame):
    # Greyscale frame
    # gray = rgb2gray(frame)

    # Crop the screen (remove the roof because it contains no information)
    # [Up: Down, Left: right]
    # cropped_frame = frame[30:-10, 30:-30]

    # Normalize Pixel Values
    # normalized_frame = cropped_frame / 255.0

    normalized_frame = frame /255.0

    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])

    # load the preprocessed frame in cv2
    image = cv2.imread(preprocessed_frame)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # create NumPy arrays from the boundaries
    lower = np.array([0, 70, 50], dtype="uint8")
    upper = np.array([10, 255, 255], dtype="uint8")

    # find the colors within the specified boundaries and apply the mask
    mask = cv2.inRange(hsv, lower, upper)

    # count number of red pixels
    count_red = np.count_nonzero(mask)
    # count total number of pixels
    total_pixels = mask.size
    # percentage red:
    red_percentage = count_red / total_pixels

    preprocessed_frame = cv2.bitwise_and(hsv, hsv, mask = mask)

    return preprocessed_frame, red_percentage

def reward(red_percentage):
    if red_percentage >= 0.10:
        return 10
    elif 0.05 >= red_percentage > 0.10 :
        return 5
    else:
        return 0


stack_size = 4  # We stack 4 frames

# Initialize deque with zero-images one array for each image
stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

def rob_action(action):
    if action == [1, 0, 0, 0, 0, 0, 0, 0]:
        rob.move(20, 20, 400)
        new_state = rob.get_image_front()
        new_frame, perc = preprocess_frame(new_state)
        reward = reward(perc)
        return reward
    elif action == [0, 1, 0, 0, 0, 0, 0, 0]:
        rob.move(2, -2, 400)
        new_state = rob.get_image_front()
        new_frame, perc = preprocess_frame(new_state)
        reward = reward(perc)
        return reward
    elif action == [0, 0, 1, 0, 0, 0, 0, 0]:
        rob.move(5, -5, 400)
        new_state = rob.get_image_front()
        new_frame, perc = preprocess_frame(new_state)
        reward = reward(perc)
        return reward
    elif action == [0, 0, 0, 1, 0, 0, 0, 0]:
        rob.move(10, -10, 400)
        new_state = rob.get_image_front()
        new_frame, perc = preprocess_frame(new_state)
        reward = reward(perc)
        return reward
    elif action == [0, 0, 0, 0, 1, 0, 0, 0]:
        rob.move(-2, 2, 400)
        new_state = rob.get_image_front()
        new_frame, perc = preprocess_frame(new_state)
        reward = reward(perc)
        return reward
    elif action == [0, 0, 0, 0, 0, 1, 0, 0]:
        rob.move(-5, 5, 400)
        new_state = rob.get_image_front()
        new_frame, perc = preprocess_frame(new_state)
        reward = reward(perc)
        return reward
    elif action == [0, 0, 0, 0, 0, 0, 1, 0]:
        rob.move(-10, 10, 400)
        new_state = rob.get_image_front()
        new_frame, perc = preprocess_frame(new_state)
        reward = reward(perc)
        return reward
    elif action == [0, 0, 0, 0, 0, 0, 0, 1]:
        rob.move(-10, -10, 400)
        new_state = rob.get_image_front()
        new_frame, perc = preprocess_frame(new_state)
        reward = reward(perc)
        return reward


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


### MODEL HYPERPARAMETERS
state_size = [84,84,4]      # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels)
action_size = 8              # 3 possible actions: left, right, shoot
learning_rate =  0.0002      # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 500        # Total episodes for training
max_steps = 100              # Max possible steps in an episode
batch_size = 64

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.0001            # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.95               # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000          # Number of experiences the Memory can keep

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False


class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            # self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.inputs_ = tf.placeholder(tf.float32, [None, 84, 84, 4], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, 3], name="actions_")

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 84x84x4
            self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                          filters=32,
                                          kernel_size=[8, 8],
                                          strides=[4, 4],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1")

            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                                 training=True,
                                                                 epsilon=1e-5,
                                                                 name='batch_norm1')

            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [20, 20, 32]

            """
            Second convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                          filters=64,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2")

            self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                                 training=True,
                                                                 epsilon=1e-5,
                                                                 name='batch_norm2')

            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
            ## --> [9, 9, 64]

            """
            Third convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                          filters=128,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")

            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                                 training=True,
                                                                 epsilon=1e-5,
                                                                 name='batch_norm3')

            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            ## --> [3, 3, 128]

            self.flatten = tf.layers.flatten(self.conv3_out)
            ## --> [1152]

            self.fc = tf.layers.dense(inputs=self.flatten,
                                      units=512,
                                      activation=tf.nn.elu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="fc1")

            self.output = tf.layers.dense(inputs=self.fc,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=3,
                                          activation=None)

            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

# Reset the graph
tf.reset_default_graph()

# Instantiate the DQNetwork
DQNetwork = DQNetwork(state_size, action_size, learning_rate)


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

# Render the environment
# rob.new_episode()
connect_VREP()

rob.play_simulation()
done = False
rob.set_phone_tilt(32, 100)
# connect to prey robot
# initialise class prey
prey_controller = prey.Prey(robot=prey_robot)
# start the thread prey, makes the prey move
prey_controller.start()

for i in range(pretrain_length):
    # If it's the first step
    if i == 0:
        # First we need a state
        state = rob.get_image_front().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    # Random action
    action = random.choice(possible_actions)

    # Get the rewards
    # reward = rob.move(action)
    reward = rob_action(action)

    # Look if the episode is finished
    # done = rob.is_episode_finished()

    # # If we're dead
    if done:
        # We finished the episode
        next_state = np.zeros(state.shape)

        # Add experience to memory
        memory.add((state, action, reward, next_state, done))

        # Start a new episode
        # rob.new_episode()
        connect_VREP()

        rob.play_simulation()
        done = False
        rob.set_phone_tilt(32, 100)
        # connect to prey robot
        # initialise class prey
        prey_controller = prey.Prey(robot=prey_robot)
        # start the thread prey, makes the prey move
        prey_controller.start()

        # First we need a state
        state = rob.get_image_front()

        # Stack the frames
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    else:
        # Get the next state
        next_state = rob.get_image_front()
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

        # Add experience to memory
        memory.add((state, action, reward, next_state))

        # Our state is now the next_state
        state = next_state

# Setup TensorBoard Writer
writer = tf.summary.FileWriter("/tensorboard/dqn/1")

## Losses
tf.summary.scalar("Loss", DQNetwork.loss)

write_op = tf.summary.merge_all()




def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = random.choice(possible_actions)

    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, state.shape))})

        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]

    return action, explore_probability


# Saver will help us to save our model
saver = tf.train.Saver()

if training == True:
    with tf.Session() as sess:
        # Initialize the variables
        sess.run(tf.global_variables_initializer())

        # Initialize the decay rate (that will use to reduce epsilon)
        decay_step = 0

        # Init the game
        # rob.play_simulation()

        connect_VREP()


        for episode in range(total_episodes):
            # Set step to 0
            step = 0

            # Initialize the rewards of the episode
            episode_rewards = []

            # Make a new episode and observe the first state
            # rob.new_episode()
            rob.play_simulation()
            done = False
            rob.set_phone_tilt(32, 100)
            # connect to prey robot
            # initialise class prey
            prey_controller = prey.Prey(robot=prey_robot)
            # start the thread prey, makes the prey move
            prey_controller.start()

            state = rob.get_image_front()

            # Remember that stack frame function also call our preprocess function.
            state, stacked_frames = stack_frames(stacked_frames, state, True)

            while step < max_steps:
                step += 1

                # Increase decay_step
                decay_step += 1

                # Predict the action to take and take it
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state,
                                                             possible_actions)

                # Do the action
                # reward = rob.move(action)
                reward = rob_action(action)

                # Look if the episode is finished
                # done = rob.is_episode_finished()

                # Add the reward to total reward
                episode_rewards.append(reward)

                # If the game is finished
                if done:
                    # the episode ends so no next state
                    next_state = np.zeros((84, 84), dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # Set step = max_steps to end the episode
                    step = max_steps

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode),
                          'Total reward: {}'.format(total_reward),
                          'Training loss: {:.4f}'.format(loss),
                          'Explore P: {:.4f}'.format(explore_probability))

                    memory.add((state, action, reward, next_state, done))

                else:
                    # Get the next state
                    next_state = rob.get_image_front().screen_buffer

                    # Stack the frame of the next_state
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # Add experience to memory
                    memory.add((state, action, reward, next_state, done))

                    # st+1 is now our current state
                    state = next_state

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

with tf.Session() as sess:
    rob, possible_actions = create_environment()

    totalScore = 0

    # Load the model
    saver.restore(sess, "./week4/models/model.ckpt")

    connect_VREP()

    for i in range(1):

        # rob.new_episode()
        rob.play_simulation()
        done = False
        rob.set_phone_tilt(32, 100)
        # connect to prey robot
        # initialise class prey
        prey_controller = prey.Prey(robot=prey_robot)
        # start the thread prey, makes the prey move
        prey_controller.start()

        while not rob.is_episode_finished():
            frame = rob.get_image_front().screen_buffer
            state = stack_frames(stacked_frames, frame)
            # Take the biggest Q value (= the best action)
            Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, state.shape))})
            action = np.argmax(Qs)
            action = possible_actions[int(action)]
            # rob.move(action)
            rob_action(action)
            score = rob.get_total_reward()
        print("Score: ", score)
        totalScore += score
        rob.stop_world()

    print("TOTAL_SCORE", totalScore / 100.0)








#
# # Initialize the CNN
# classifier = Sequential()
#
# # Step 1 - Convolution
# #number of filters (32), shape for each filter (3, 3), input shape (64, 64), type of image(RGB(3) or B/W), activation function
# classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = "relu"))
#
# # Step 2 - Pooling (dit moet er dus uit!!!)
# classifier.add(MaxPooling2D(pool_size = (2, 2)))
#
# # Step 3 - Flattening
# classifier.add(Flatten())
#
# # Step 4 - Full Connection
# classifier.add(Dense(activation = "relu", units = 128)) #output_dim = 128
# classifier.add(Dense(activation = "softmax", units = 6)) #output_dim = 6
#
# # Compiling the CNN
# classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#
# # Fitting the CNN to the images
# train_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# training_set = train_datagen.flow_from_directory(
#     'week3/images/dataset/training_set',
#     target_size = (64, 64),
#     batch_size = 32, #Number of observations per batch
#     class_mode = 'categorical'
# )
#
# test_set = test_datagen.flow_from_directory(
#     'week3/images/dataset/test_set',
#     target_size = (64, 64),
#     batch_size = 32, #Number of observations per batch
#     class_mode = 'categorical'
# )
#
# # Checkpoint
# filepath = "week3/models/CNN_Sim_weights(11){epoch:02d}-{val_loss:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
# callbacks_list = [checkpoint]
#
# # Train convolutional neural network
# history = classifier.fit_generator(
#     training_set,
#     steps_per_epoch = 661, #Number of training images
#     epochs = 10, #1 epoch means neural network is trained on every training examples in 1 pass --> training cycle
#     validation_data = test_set,
#     validation_steps = 5, #suggestion: validation_steps = TotalvalidationSamples / ValidationBatchSize
#     callbacks = callbacks_list
# )
#
# # Get training and test loss histories
# training_loss = history.history['loss']
# test_loss = history.history['val_loss']
#
# # Get training and test loss histories
# training_acc = history.history['acc']
# test_acc = history.history['val_acc']
#
# # Create count of the number of epochs
# epoch_count = range(1, len(training_loss) + 1)
#
# # Visualize loss history
# fig_Loss = plt.figure()
# plt.plot(epoch_count, training_loss, 'r--')
# plt.plot(epoch_count, test_loss, 'b-')
# plt.legend(['Training Loss', 'Test Loss'])
# fig_Loss.suptitle('Loss History', fontsize=18)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show();
# fig_Loss.savefig('week3/figures/fig_Sim_LossHistory(11).png')
#
# # Visualize accuracy history
# fig_Acc = plt.figure()
# plt.plot(epoch_count, training_acc, 'r--')
# plt.plot(epoch_count, test_acc, 'b-')
# plt.legend(['Training Accuracy', 'Test Accuracy'])
# fig_Acc.suptitle('Accuracy History', fontsize=18)
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.show();
# fig_Acc.savefig('week3/figures/fig_Sim_AccHistory(11).png')
#
# # serialize model to JSON
# # the keras model which is trained is defined as 'model' in this example
# model_json = classifier.to_json()
# with open("week3/models/CNN_model(11).json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# classifier.save_weights("week3/models/CNN_weights(11).h5")
# print("Saved model to disk")
#
# # Test a random image
# # Go 45 right
# test_image = image.load_img('week3/images/p2_fail/img_p2-190.png', target_size = (64, 64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0) # Add fourth dimension
# # Go straight
# test_image2 = image.load_img('week3/images/p2_fail/img_p2-1.png', target_size = (64, 64))
# test_image2 = image.img_to_array(test_image2)
# test_image2 = np.expand_dims(test_image2, axis = 0)
# # Go backward
# test_image3 = image.load_img('week3/images/p2_fail/img_p2-36.png', target_size = (64, 64))
# test_image3 = image.img_to_array(test_image3)
# test_image3 = np.expand_dims(test_image3, axis = 0)
#
# result = classifier.predict(test_image)
# result2 = classifier.predict(test_image2)
# result3 = classifier.predict(test_image3)
#
# print(result)
# print(result2)
# print(result3)
#
# training_set.class_indices
#
# if result[0][0] == 1:
#     prediction = 'straight'
# elif result[0][1] == 1 :
#     prediction = '45right'
# elif result[0][2] == 1:
#     prediction = '90right'
# elif result[0][3] == 1:
#     prediction = '45left'
# elif result[0][4] == 1:
#     prediction = '90left'
# elif result[0][5] == 1:
#     prediction = 'back'
#
# if result2[0][0] == 1:
#     prediction2 = 'straight'
# elif result2[0][1] == 1 :
#     prediction2 = '45right'
# elif result2[0][2] == 1:
#     prediction2 = '90right'
# elif result2[0][3] == 1:
#     prediction2 = '45left'
# elif result2[0][4] == 1:
#     prediction2 = '90left'
# elif result2[0][5] == 1:
#     prediction2 = 'back'
#
# if result3[0][0] == 1:
#     prediction3 = 'straight'
# elif result3[0][1] == 1 :
#     prediction3 = '45right'
# elif result3[0][2] == 1:
#     prediction3 = '90right'
# elif result3[0][3] == 1:
#     prediction3 = '45left'
# elif result3[0][4] == 1:
#     prediction3 = '90left'
# elif result3[0][5] == 1:
#     prediction3 = 'back'
#
# print(prediction)
# print(prediction2)
# print(prediction3)
