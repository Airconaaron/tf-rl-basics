import numpy as np 
import random
import gym
import tensorflow as tf

import cv2

from network import *

EPSILON = 0.05 # test exploration parameter

env = gym.make('Breakout-v0')
my = DQN(EPSILON, actions=4)

def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110,:]
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(84,84,1))


for i_episode in range(200):
    observation = env.reset()
    my.initState(my.preprocess(observation))
    for t in range(100):
        #env.render()
        #print(observation)
        #print(observation.shape) 210, 160, 3 (shape of the input)
        #print(env.action_space.n) #4 actions 
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action) # if we ever go beyond this we'll just sample from a random point # info tells us stuff about lives and stuff

        print(action, "time", i_episode*100 + t)
        
        obj = preprocess(observation)        
            #print(obj3, obj3.shape)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break


# grab our saved model parameters