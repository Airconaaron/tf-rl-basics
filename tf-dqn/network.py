import tensorflow as tf 
import numpy as np 
import random
from collections import deque 


# Hyper Parameters As Specified in Paper:
BATCH_SIZE = 32
REPLAY_MEMORY = 1000000
AGENT_HISTORY = 4 # number of frames in the past our agent sees
TARGET_UPDATE_FREQ = 10000
GAMMA = 0.99
ACTION_REPEAT = 4 # for how many frames do we repeat actions
UPDATE_FREQ = 4

LEARNING_RATE = 0.00025
GRADIENT_MOMENTUM = 0.95
SQUARED_GRAD = 0.95
MIN_SQUARED_GRAD = 0.01

FINAL_EPSILON = 0.1#0.001 # final value of epsilon
INITIAL_EPSILON = 1.0#0.01 # starting value of epsilon
EXPLORE_FRAMES = 1000000. # frames over which to anneal epsilon
OBSERVE = 50000. # timesteps to observe before training through experience replay
NO_OP_MAX = 30 # number of max frames before we start doing actions
ATARI_NUM = 16 # which is up, down, left, righ diagonals times two because of the button


# epsilon determines our exploration trade off
class DQN:
    def __init__(self, init_epsilon=INITIAL_EPSILON, actions=ATARI_NUM):
        self.memory = deque()
        self.timeStep = 0
        self.epsilon = init_epsilon
        self.actions = actions # atari defaults
        self.prevAction = 0

        # notice that there really isn't any dropout
        with tf.name_scope('normal'):
            # input size of 4 frames 
            self.input = tf.placeholder("float", [None, 84, 84, AGENT_HISTORY], name = "input") # our image must be converted to greyscale first

            self.weight1_conv = tf.Variable(tf.truncated_normal(shape=[8,8,AGENT_HISTORY, 32], stddev=0.01), name="weight1_conv")
            self.bias1_conv = tf.Variable(tf.constant(0.01, shape=[32]), name="bias1_conv")
            self.conv1 = tf.nn.relu(tf.nn.conv2d(self.input, self.weight1_conv, strides=[1,4,4,1], padding="VALID") + self.bias1_conv, name = 'conv1')

            self.weight2_conv = tf.Variable(tf.truncated_normal(shape=[4,4,32,64], stddev=0.01), name="weight2_conv")
            self.bias2_conv = tf.Variable(tf.constant(0.01, shape=[64]), name="bias2_conv")
            self.conv2 = tf.nn.relu(tf.nn.conv2d(self.conv1, self.weight2_conv, strides=[1,2,2,1], padding="VALID") + self.bias2_conv, name="conv2")

            self.weight3_conv = tf.Variable(tf.truncated_normal(shape=[3,3,64,64], stddev=0.01), name="weight3_conv")
            self.bias3_conv = tf.Variable(tf.constant(0.01, shape=[64]), name="bias3_conv")
            self.conv3 = tf.nn.relu(tf.nn.conv2d(self.conv2, self.weight3_conv, strides=[1,1,1,1], padding="VALID") + self.bias3_conv, name="conv3")
            conv3_flat = tf.reshape(self.conv3, [-1, 7*7*64])

            self.weight4_fc = tf.Variable(tf.truncated_normal(shape=[3136, 512], stddev=0.01), name="weight4_fc")
            self.bias4_fc = tf.Variable(tf.constant(0.01, shape=[512]), name="bias4_fc")
            self.fc4 = tf.nn.relu(tf.matmul(conv3_flat, self.weight4_fc) + self.bias4_fc, name="fc1")

            self.weight5_fc = tf.Variable(tf.truncated_normal(shape=[512, ATARI_NUM], stddev=0.01), name="weight5_fc")
            self.bias5_fc = tf.Variable(tf.constant(0.01, shape=[ATARI_NUM]), name="bias5_fc")
            self.QValue = (tf.matmul(self.fc4, self.weight5_fc) + self.bias5_fc)

        with tf.name_scope('target'):
            self.inputT = tf.placeholder("float", [None, 84, 84, AGENT_HISTORY], name = "inputT") # our image must be converted to greyscale first

            self.weight1_convT = tf.Variable(tf.truncated_normal(shape=[8,8,AGENT_HISTORY, 32], stddev=0.01), name="weight1_convT")
            self.bias1_convT = tf.Variable(tf.constant(0.01, shape=[32]), name="bias1_convT")
            self.conv1T = tf.nn.relu(tf.nn.conv2d(self.inputT, self.weight1_convT, strides=[1,4,4,1], padding="VALID") + self.bias1_convT, name = 'conv1T')

            self.weight2_convT = tf.Variable(tf.truncated_normal(shape=[4,4,32,64], stddev=0.01), name="weight2_convT")
            self.bias2_convT = tf.Variable(tf.constant(0.01, shape=[64]), name="bias2_convT")
            self.conv2T = tf.nn.relu(tf.nn.conv2d(self.conv1T, self.weight2_convT, strides=[1,2,2,1], padding="VALID") + self.bias2_convT, name="conv2T")

            self.weight3_convT = tf.Variable(tf.truncated_normal(shape=[3,3,64,64], stddev=0.01), name="weight3_convT")
            self.bias3_convT = tf.Variable(tf.constant(0.01, shape=[64]), name="bias3_convT")
            self.conv3T = tf.nn.relu(tf.nn.conv2d(self.conv2T, self.weight3_convT, strides=[1,1,1,1], padding="VALID") + self.bias3_convT, name="conv3T")
            self.conv3_flatT = tf.reshape(self.conv3T, [-1, 7*7*64])

            self.weight4_fcT = tf.Variable(tf.truncated_normal(shape=[3136, 512], stddev=0.01), name="weight4_fcT")
            self.bias4_fcT = tf.Variable(tf.constant(0.01, shape=[512]), name="bias4_fcT")
            self.fc4T = tf.nn.relu(tf.matmul(self.conv3_flatT, self.weight4_fcT) + self.bias4_fcT, name="fc1T")

            self.weight5_fcT = tf.Variable(tf.truncated_normal(shape=[512, ATARI_NUM], stddev=0.01), name="weight5_fcT")
            self.bias5_fcT = tf.Variable(tf.constant(0.01, shape=[ATARI_NUM]), name="bias5_fcT")
            self.QValueT = (tf.matmul(self.fc4T, self.weight5_fcT) + self.bias5_fcT)


        self.copy_tensors = [
            tf.assign(self.weight1_convT, self.weight1_conv), 
            tf.assign(self.bias1_convT, self.bias1_conv), 
            tf.assign(self.weight2_convT, self.weight2_conv),
            tf.assign(self.bias2_convT, self.bias2_conv),
            tf.assign(self.weight3_convT, self.weight3_conv),
            tf.assign(self.bias3_convT, self.bias3_conv),
            tf.assign(self.weight4_fcT, self.weight4_fc),
            tf.assign(self.bias4_fcT, self.bias4_fc),
            tf.assign(self.weight5_fcT, self.weight5_fc),
            tf.assign(self.bias5_fcT, self.bias5_fc)
            ]
        # define a operation to copy over our 'normal' network to the target

        self.create_placeholder()

        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def copy_to_target(self):
        self.sess.run(self.copy_tensors)

    def create_placeholder(self):
        self.action_input = tf.placeholder("float", [None, ATARI_NUM], name="Action_Input")
        self.yInput = tf.placeholder("float", [None])
        Q_Action = tf.reduce_sum(tf.matmul(self.QValue, self.action_input), reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
        self.trainStep = tf.train.RMSPropOptimizer(LEARNING_RATE,0.99, GRADIENT_MOMENTUM,1e-6).minimize(self.cost)

    def train_network(self):
        minibatch = random.sample(self.memory,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]

        # Step 2: calculate y 
        y_batch = []
        QValue_batch = self.QValueT.eval(feed_dict={self.inputT:nextState_batch})
        for i in range(0,BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

        self.trainStep.run(feed_dict={
            self.yInput : y_batch,
            self.action_input : action_batch,
            self.input : state_batch
            })

        # save network every 100000 iteration
        if self.timeStep % 10000 == 0:
            save_file = "./model/netowrk-dqn" + str(self.timeStep) 
            self.saver.save(self.sess, save_file)

        if self.timeStep % UPDATE_TIME == 0:
            self.copy_to_target()

    def set_perception(self,obsv, action, reward, terminal):
        newState = np.append(obsv,self.currentState[:,:,1:],axis = 2)
        self.memory.append((self.currentState,action,reward,newState,terminal))

        if len(self.memory) > REPLAY_MEMORY:
            self.memory.popleft()
        if self.timeStep > OBSERVE:
            # Train the network
            self.train_network()

        # print stuff later
        state = ""
        if self.timeStep <= OBSERVE:
            state = "observe"
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE_FRAMES:
            state = "explore"
        else:
            state = "train"

        print ("TIMESTEP", self.timeStep, "/ STATE", state, \
        "/ EPSILON", self.epsilon)
        self.currentState = newState
        self.timeStep += 1

    def get_action(self):
        QValue = self.sess.run(self.QValue, feed_dict= {self.input: [self.currentState]})
        action = self.prevAction

        # time to change
        if self.timeStep % ACTION_REPEAT  == 0:
            if random.random() <= self.epsilon:
                action = random.randrange(0, self.actions)
            else:
                action = np.argmax(QValue)
        # change episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE_FRAMES

        return action

    def initState(self, observation):
        self.currentState = np.stack([observation for i in range(AGENT_HISTORY)], axis = 2)
        #self.currentState = self.currentState[84,84, AGENT_HISTORY]
        self.currentState = np.reshape(self.currentState, [84, 84 , AGENT_HISTORY])

    # method to create a 84,84 greyscale image
    def preprocess(self, observation):
        obj1 = tf.image.rgb_to_grayscale(observation)
        obj2 = tf.image.resize_images(obj1, [84,84])
        obj3 = self.sess.run(obj2)

        #return tf.reshape(obj3, [-1, 84, 84, 1] )
        return obj3




