# main defintion for the reinforcement learning model

# Imports
import numpy as np
import tensorflow as tf
import pickle
import statistics
import tqdm
import matplotlib.pyplot as plt
import copy
import datetime
import os

# Custom Imports
from model.ActorCritic import CutActorCritic, RandomSelector
from model.Utils import *
from model.Environments import CutEnvironment

######## Parameters ########
seed = 324 # seed for numpy and tensorflow

circ_filename = "../../qcircml_code/data/circol_test.p" # filename of circuit collection

# batch parameters
batch_size = 30
loops = 110
train_percent = 0.8

# model parameters
action_size = 6 # number of actions the agent can take
fc_layer_list = [512, 256, 128] # list of number of hidden units for each desired fully connected layer

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
critic_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM) # define critic loss function
load = False # load model weights from file # FIXME: not working
model_load_filename = "../../qcircml_code/data/model_weights3.h5" # filename of model weights

# training parameters
window_size = 100 # size of window for moving average

# saving parameters
save = True # save data to file
root_dir = "../../qcircml_code/data_" + datetime.datetime.now().strftime("%m%d%Y") + "/"
date_str = datetime.datetime.now().strftime("%m%d%Y") # used for saving data

######## Set Seed ########
np.random.seed(seed)
# tf.random.set_seed(seed)

# load circuit collection
circol = pickle.load(open(circ_filename, "rb"))

######## Create Batched Dataset ########
train_data, val_data = create_dataset(batch_size, loops, circol, train_percent)

print("\ntrain_data:", train_data.shape)
print("val_data:", val_data.shape)

######## Create Environment and Model ########

env = CutEnvironment(circol) # create cut environment
model = CutActorCritic(action_size, fc_layer_list) # create model
rando = RandomSelector(action_size)

if load:
    image_shape = (circol.images[-1][0].shape[0], circol.images[-1][0].shape[1])
    dummy = tf.zeros((1, image_shape[0], image_shape[1]))

    print("image_shape:", image_shape)
    print("dummy:", str(dummy))
    
    action_logits_c, values = model(dummy) # call model once to initialize weights
    model.load_weights(model_load_filename)

# test train step

# tf.summary.trace_on(graph=True, profiler=True)
# episode_reward = int(train_step(train_data[0], model, env, critic_loss, optimizer, gamma=0.99))
# writer = tf.summary.create_file_writer("../qcircml_code/data/logs")
# with writer.as_default():
#     tf.summary.trace_export(name="my_func_trace", step=0, profiler_outdir="../qcircml_code/data/logs")
# episode_reward2 = int(train_step(train_data[1], model, env, critic_loss, optimizer, gamma=0.99))

# print("episode_reward:", episode_reward)
# print("episode_reward2:", episode_reward2)

# quit()

######## Save Setup ########
# check if root_dir exists, if not create it
if not os.path.exists(root_dir):
    os.mkdir(root_dir)

# get list of all files in root_dir
files = os.listdir(root_dir)

# make list of substring of all filenames after last underscore
runs = [int(filename.split("_")[-1]) for filename in files] # this is the run number
max_run = max(runs) if len(runs) > 0 else -1 # get max run number

######## Train Model ########
model_save_filename = root_dir + date_str + "_model_weights_" + str(max_run + 1) + ".h5"
episode_rewards, random_rewards, running_average, random_average = train_loop(train_data, model, rando, env, critic_loss, optimizer, window_size, model_save_filename)

######## Save Data ########
csv_filename = root_dir + date_str + "_data_" + str(max_run + 1) + ".csv"
# put all data into one csv file
# data = np.array([episode_rewards, random_rewards, running_average, random_average])
# np.savetxt("../qcircml_code/data/data.csv", data, delimiter=",")

######## Plot Data ########
plot_filename = root_dir + date_str + "_plot_" + str(max_run + 1) + ".png"

print()
# print("Possible Rewards:",list(set(episode_rewards)))
# ep_avg = statistics.mean(list(set(episode_rewards)))
# print("Average of Possible Rewards:", ep_avg)

# plot episode rewards
plt.title("Episode Rewards")
# plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Episode Reward")

# plot moving average
moving_average = []
moving_avg_random = []
for i in range(len(episode_rewards) - 1):
    moving_average.append(statistics.mean(episode_rewards[max(0, i - window_size):i + 1]))
    moving_avg_random.append(statistics.mean(random_rewards[max(0, i - window_size):i + 1]))

plt.plot(moving_average, color='k', label='Agent')
plt.plot(moving_avg_random, color='b', label='Random')
# show horizontal line at average
plt.axhline(y=random_average, color='r', linestyle='-', label='Random Average')
plt.legend()

fig = plt.gcf()
fig.savefig(plot_filename)
plt.show()