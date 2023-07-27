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
from model.Environments import CutEnvironment

# setting seed
seed = 324
np.random.seed(seed)
tf.random.set_seed(seed)

# load circuit collection
circol = pickle.load(open("../qcircml_code/data/circol_test.p", "rb"))

# generate images
circol.convert_to_images()

# create batches
batch_size = 5
loops = 5
batched_circuits = []
# l = np.arange(0, len(circol.circuits[-1]))
# l = np.arange(0, 10)
index_list_master = [[len(circol.circuits) - 1, i] for i in range(0, len(circol.circuits[-1])) ] # create list of all possible circuit indexes

# put 80% of the data in the training set
index_list = index_list_master[:int(len(index_list_master) * 0.02)]
validation_list = index_list_master[int(len(index_list_master) * 0.8):]

for i in range(loops):

    l = copy.deepcopy(index_list)
    np.random.shuffle(l) # shuffle list
    batched_circuits_temp = [l[i:i + batch_size] for i in range(0, len(l), batch_size)]

    # remove last batch if it is not full
    if len(batched_circuits_temp[-1]) != batch_size:
        batched_circuits_temp.pop(-1)

    batched_circuits.extend(batched_circuits_temp)

# 80% of the data is used for training, 20% for validation
train_data = batched_circuits

# convert to tensor
train_data = tf.convert_to_tensor(train_data, dtype=tf.int32)
print("train_data:",train_data.shape)

# print the number of training and validation batches
print("Number of training batches: " + str(len(train_data)))
# print("Number of validation batches: " + str(len(val_data)))

# create optimizer
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)

# create cut environment
env = CutEnvironment(circol)

# define critic loss function
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

# create model
model = CutActorCritic(6, [512, 256, 128])
load = False

# test train step
episode_reward = int(train_step(train_data[0], model, env, huber_loss, optimizer, gamma=0.99))
episode_reward2 = int(train_step(train_data[1], model, env, huber_loss, optimizer, gamma=0.99))

print("episode_reward:", episode_reward)
print("episode_reward2:", episode_reward2)

# # training loop
# episode_rewards = []

# t = tqdm.trange(len(train_data))
# for i in t:
#     # run train step
#     episode_reward = int(train_step(tf.convert_to_tensor(train_data[i]), model, circol, optimizer, gamma=0.99))

#     if i == 0 and load:
#         model.load_weights("../qcircml_code/data/model_weights3.h5")

#     # store episode reward
#     episode_rewards.append(episode_reward)

#     # keep running average of episode rewards
#     running_average = statistics.mean(episode_rewards)

#     # calculate average of last 100 episodes
#     if i > 100:
#         moving_average = statistics.mean(episode_rewards[i - 100:i])
#     else:
#         moving_average = running_average

#     # update tqdm
#     t.set_description("Running average: " + str(round(running_average,2)) + " Moving average: " + str(round(moving_average,2)))

# print()
# print("Possible Rewards:",list(set(episode_rewards)))
# ep_avg = statistics.mean(list(set(episode_rewards)))
# print("Average of Possible Rewards:", ep_avg) 

# # save model
# model.save_weights("../qcircml_code/data/model_weights3.h5")

# # plot episode rewards
# plt.plot(episode_rewards)
# plt.xlabel("Episode")
# plt.ylabel("Episode Reward")

# # plot moving average
# moving_average = []
# for i in range(len(episode_rewards) - 1):
#     moving_average.append(statistics.mean(episode_rewards[max(0, i - 100):i + 1]))

# plt.plot(moving_average, color='g')
# # show horizontal line at average
# plt.axhline(y=ep_avg, color='r', linestyle='-')
# plt.show()