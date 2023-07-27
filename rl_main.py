# main defintion for the reinforcement learning model
#

# Imports
import numpy as np
import tensorflow as tf
import pickle
import statistics
import tqdm
import matplotlib.pyplot as plt

import copy

# Custom Imports
from model.ActorCritic import CutActorCritic
from model.Utils import *

# globals
seed = 324
np.random.seed(seed)
tf.random.set_seed(seed)

# enable eager execution
tf.compat.v1.enable_eager_execution()

# create test circuit batch
# circuit_batch = tf.convert_to_tensor(np.array([[1, 10], [1, 11]]))

# create model
model = CutActorCritic(6, 512)
load = False

# load circuit collection
circol = pickle.load(open("./data/circol_test.p", "rb"))

# generate images
circol.convert_to_images()

# create batches
batch_size = 30
loops = 100
batched_circuits = []
l = np.arange(0, len(circol.circuits[-1]))
index_list_master = [[len(circol.circuits) - 1, i] for i in range(0, len(circol.circuits[-1])) ] # create list of all possible circuit indexes

# put 80% of the data in the training set
index_list = index_list_master[:int(len(index_list_master) * 0.8)]
validation_list = index_list_master[int(len(index_list_master) * 0.8):]

for i in range(loops):

    l = copy.deepcopy(index_list)
    np.random.shuffle(l) # shuffle list
    batched_circuits_temp = [l[i:i + batch_size] for i in range(0, len(l), batch_size)] 

    batched_circuits.extend(batched_circuits_temp)

# 80% of the data is used for training, 20% for validation
train_data = batched_circuits

# train_data = batched_circuits[:int(len(batched_circuits) * 0.8)]
# val_data = batched_circuits[int(len(batched_circuits) * 0.8):]

# print the number of training and validation batches
print("Number of training batches: " + str(len(train_data)))
# print("Number of validation batches: " + str(len(val_data)))

# create optimizer
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)

# test train step
episode_reward = int(train_step(tf.convert_to_tensor(train_data[0]), model, circol, optimizer, gamma=0.99))
print("Episode reward: " + str(episode_reward))
quit()

# training loop
episode_rewards = []

t = tqdm.trange(len(train_data))
for i in t:
    # run train step
    episode_reward = int(train_step(tf.convert_to_tensor(train_data[i]), model, circol, optimizer, gamma=0.99))

    if i == 0 and load:
        model.load_weights("./data/model_weights2.h5")

    # store episode reward
    episode_rewards.append(episode_reward)

    # keep running average of episode rewards
    running_average = statistics.mean(episode_rewards)

    # calculate average of last 100 episodes
    if i > 100:
        moving_average = statistics.mean(episode_rewards[i - 100:i])
    else:
        moving_average = running_average

    # update tqdm
    t.set_description("Running average: " + str(round(running_average,2)) + " Moving average: " + str(round(moving_average,2)))

print()
print("Possible Rewards:",list(set(episode_rewards)))
ep_avg = statistics.mean(list(set(episode_rewards)))
print("Average of Possible Rewards:", ep_avg) 

# save model
model.save_weights("./data/model_weights_07262023.h5")

# plot episode rewards
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Episode Reward")

# plot moving average
moving_average = []
for i in range(len(episode_rewards) - 1):
    moving_average.append(statistics.mean(episode_rewards[max(0, i - 100):i + 1]))

plt.plot(moving_average, color='g')
# show horizontal line at average
plt.axhline(y=ep_avg, color='r', linestyle='-')
plt.show()