import gym
import csv
import gym_pull
from random import randint
import time
import numpy as np
import h5py
from matplotlib import pyplot as plt
from keras.utils import np_utils
import keras.callbacks as cb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.datasets import mnist
from numpy import genfromtxt
from keras.models import model_from_json


#Cargar modelo 
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

for i_episode in range(1000):
	#string = 'ppaquette/SuperMarioBros-{}-{}-v0'.format(randint(1, 8), randint(1, 4))
	string = 'ppaquette/SuperMarioBros-{}-{}-v0'.format(1,1)
	env = gym.make(string)
	observation = env.reset()
	distance=0
	while True:
		env.render()
		action = env.action_space.sample()
		# 0: [0, 0, 0, 0, 0, 0],  # NOOP
		# 1: [1, 0, 0, 0, 0, 0],  # Up
		# 2: [0, 0, 1, 0, 0, 0],  # Down
		# 3: [0, 1, 0, 0, 0, 0],  # Left
		# 4: [0, 1, 0, 0, 1, 0],  # Left + A
                # 5: [0, 1, 0, 0, 0, 1],  # Left + B
                # 6: [0, 1, 0, 0, 1, 1],  # Left + A + B
                # 7: [0, 0, 0, 1, 0, 0],  # Right
                # 8: [0, 0, 0, 1, 1, 0],  # Right + A
                # 9: [0, 0, 0, 1, 0, 1],  # Right + B
                # 10: [0, 0, 0, 1, 1, 1],  # Right + A + B
                # 11: [0, 0, 0, 0, 1, 0],  # A
                # 12: [0, 0, 0, 0, 0, 1],  # B
                # 13: [0, 0, 0, 0, 1, 1],  # A + B
            	old_observation = observation
		x = np.append(old_observation, [])
                try:
		    action_v = model.predict(np.array([x]), batch_size=16, verbose=1)
                    action = action_v.astype(int)[0].tolist()
                    print action
		    observation, reward, done, info = env.step(action)
		except Exception as e:
		    print e
		    observation, reward, done, info = env.step(action)
            	if done:
                	break
