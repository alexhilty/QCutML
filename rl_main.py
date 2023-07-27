# main defintion for the reinforcement learning model

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

######## Parameters ########
seed = 324 # seed for numpy and tensorflow

circ_filename = "../qcircml_code/data/circol_test.p" # filename of circuit collection

# batch parameters
batch_size = 30
loops = 100
train_percent = 0.8

# model parameters
action_size = 6 # number of actions the agent can take
fc_layer_list = [512, 256, 128] # list of number of hidden units for each desired fully connected layer

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
critic_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM) # define critic loss function
load = False # load model weights from file # FIXME: not working
model_filename = "../qcircml_code/data/model_weights3.h5" # filename of model weights

# training parameters
window_size = 100 # size of window for moving average

######## Set Seed ########
np.random.seed(seed)
tf.random.set_seed(seed)

# load circuit collection
circol = pickle.load(open(circ_filename, "rb"))

######## Create Batched Dataset ########
train_data, val_data = create_dataset(batch_size, loops, circol, train_percent)

print("\ntrain_data:", train_data.shape)
print("val_data:", val_data.shape)


######## Create Environment and Model ########

env = CutEnvironment(circol) # create cut environment
model = CutActorCritic(action_size, fc_layer_list) # create model

# test train step
# episode_reward = int(train_step(train_data[0], model, env, critic_loss, optimizer, gamma=0.99))
# episode_reward2 = int(train_step(train_data[1], model, env, critic_loss, optimizer, gamma=0.99))

# print("episode_reward:", episode_reward)
# print("episode_reward2:", episode_reward2)

# training loop
episode_rewards = []

t = tqdm.trange(len(train_data)) # for showing progress bar
for i in t:
    # run train step
    episode_reward = int(train_step(train_data[i], model, env, critic_loss, optimizer, gamma=0.99))

    # if i == 0 and load:
    #     model.load_weights("../qcircml_code/data/model_weights3.h5")

    # store episode reward
    episode_rewards.append(episode_reward)

    # keep running average of episode rewards
    running_average = statistics.mean(episode_rewards)

    # calculate average of last 100 episodes
    if i > window_size:
        moving_average = statistics.mean(episode_rewards[i - 100:i])
    else:
        moving_average = running_average

    # update tqdm (progress bar)
    t.set_description("Running average: {:04.2f}, Moving average: {:04.2f}".format(running_average, moving_average))

print()
print("Possible Rewards:",list(set(episode_rewards)))
ep_avg = statistics.mean(list(set(episode_rewards)))
print("Average of Possible Rewards:", ep_avg) 

# save model
model.save_weights("../qcircml_code/data/model_weights3.h5")

# plot episode rewards
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Episode Reward")

# plot moving average
moving_average = []
for i in range(len(episode_rewards) - 1):
    moving_average.append(statistics.mean(episode_rewards[max(0, i - window_size):i + 1]))

plt.plot(moving_average, color='g')
# show horizontal line at average
plt.axhline(y=ep_avg, color='r', linestyle='-')
plt.show()