import numpy as np 
import random
import gym
import tensorflow as tf

from network import *

EPSILON = 0.05 # test exploration parameter

env = gym.make('Breakout-v0')
action_num = env.action_space.n
MyNetwork = DQN(EPSILON, actions=action_num,game="BREAKOUT")
MyNetwork.load()
NUM_EPOCHS = 100

for i in range(NUM_EPOCHS):
    observation = env.reset()
    MyNetwork.initState(preprocess(observation))
    for t in range(5*60*60):
        #env.render()
        action = MyNetwork.get_action()
        observation, reward, done, info = env.step(action)
        observation = preprocess(observation)
        MyNetwork.set_perception(observation, action, reward, done)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print("Rewards were: {}".format(MyNetwork.rewards))
            MyNetwork.done_writer(i)
            break
