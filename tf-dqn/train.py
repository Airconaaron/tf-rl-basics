import numpy as np
import random
import gym

from network import *


NUM_EPOCHS = 1000

env = gym.make('Breakout-v0')
action_num = env.action_space.n
MyNetwork = DQN(actions=action_num)

for i in range(NUM_EPOCHS):
    observation = env.reset()
    MyNetwork.initState(preprocess(observation))
    for t in range(5*60*60):
        #env.render()
        action = MyNetwork.get_action()
        observation, reward, done, info = env.step(action)
        observation = preprocess(observation)
        MyNetwork.set_perception(observation, action, reward, done)
        
        tf.summary.scalar('rewards each time', reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
