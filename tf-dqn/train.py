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
    MyNetwork.initState(MyNetwork.preprocess(observation))
    reward_val = 0
    for t in range(5*60*60):
        #env.render()
        action = MyNetwork.get_action()
        observation, reward, done, info = env.step(action)
        observation = MyNetwork.preprocess(observation)
        MyNetwork.set_perception(observation, action, reward, done)
        reward_val += reward 
        
        tf.summary.scalar('rewards each time', reward)
        if done:
            print("Episode finished after {} timesteps with reward {}".format(t+1, reward_val))
            break
