import numpy as np 
import random
import gym
import tensorflow as tf

from network import *

EPSILON = 0.05 # test exploration parameter

env = gym.make('Breakout-v0')
my = DQN(EPSILON, actions = 4)

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        #print(observation)
        #print(observation.shape) 210, 160, 3 (shape of the input)
        print(env.action_space.n) #4 actions 
        action = env.action_space.sample()
        # print(action)
        observation, reward, done, info = env.step(action) # if we ever go beyond this we'll just sample from a random point # info tells us stuff about lives and stuff
        #print(observation, observation.shape)
        
        obj3 = my.preprocess(observation)
        print(obj3.shape)
            # enc = tf.image.encode_jpeg(obj2)
            # fname = tf.constant("img/" + str(t) + 'photo.jpg')
            #fwrite = sess.run(tf.write_file(fname, enc))
            #list_obj = [obj2 for i in range(4)]
            #obj3 = sess.run(tf.reshape(tf.stack(list_obj, axis=2),[-1, 84, 84, 4]))

            #print(obj3, obj3.shape)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break


# grab our saved model parameters