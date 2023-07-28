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
import csv

# Custom Imports
from model.ActorCritic import CutActorCritic, RandomSelector
from model.Utils import *
from model.Environments import CutEnvironment

######## Parameters ########
seed = 324 # seed for numpy and tensorflow

circ_filename = "../../qcircml_code/data/circol_test.p" # filename of circuit collection

# batch parameters
batch_size = 30
loops = 200
train_percent = 0.8

# model parameters
action_size = 6 # number of actions the agent can take
fc_layer_list = [512, 256, 128, 64] # list of number of hidden units for each desired fully connected layer

learning_rate = 0.01 # learning rate for optimizer
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
critic_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM) # define critic loss function
load = False # load model weights from file
model_load_filename = "../../qcircml_code/data/model_weights3.h5" # filename of model weights

# training parameters
window_size = 100 # size of window for moving average

# saving parameters
save = True # save data to file
root_dir = "../../qcircml_code/data_" + datetime.datetime.now().strftime("%m%d%Y") + "_2/"
date_str = datetime.datetime.now().strftime("%m%d%Y") # used for saving data

# put all parameters into a dictionary
parameters = {
    "seed": seed,
    "circ_filename": circ_filename,
    "batch_size": batch_size,
    "loops": loops,
    "train_percent": train_percent,
    "action_size": action_size,
    "fc_layer_list": fc_layer_list,
    "learning_rate": learning_rate,
    "optimizer": type(optimizer),
    "critic_loss": type(critic_loss),
    "load": load,
    "model_load_filename": model_load_filename,
    "window_size": window_size,
    "save": save,
    "root_dir": root_dir,
    "date_str": date_str
}

######## Set Seed ########
np.random.seed(seed)
# tf.random.set_seed(seed)

# load circuit collection
circol = pickle.load(open(circ_filename, "rb"))

######## Create Batched Dataset ########
train_data, train_index, val_data = create_dataset(batch_size, loops, circol, train_percent)

# print("train_data:", train_data.shape)
# print("val_data:", val_data.shape)

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
runs = [int(filename.split("_")[-2]) for filename in files] # this is the run number
max_run = max(runs) if len(runs) > 0 else -1 # get max run number

######## Save Parameters ########
parameters_filename = root_dir + date_str + "_" + str(max_run + 1) + "_parameters"  + ".txt"

# save parameters to file
with open(parameters_filename, 'w') as f:
    for key in parameters.keys():
        f.write("%s: %s\n" % (key, parameters[key]))

# pickle parameter dictionary
parameters_pkl_filename = root_dir + date_str + "_" + str(max_run + 1) + "_parameters"  + ".p"
pickle.dump(parameters, open(parameters_pkl_filename, "wb"))

######## Train Model ########
model_save_filename = root_dir + date_str + "_" + str(max_run + 1) + "_weights" + ".h5"
episode_rewards, random_rewards, running_average, random_average = train_loop(train_data, model, rando, env, critic_loss, optimizer, window_size, model_save_filename)

######## Validate Model ########
hist_filename = root_dir + date_str + "_" + str(max_run + 1) + "_hist" + ".txt"

optimal_cuts, optimal_circuits_index = compute_best_cuts(circol)
chosen_cuts, hist = validation(val_data, model, env, optimal_cuts)
random_cuts, random_hist = validation(val_data, rando, env, optimal_cuts)

chosen_cuts_t, hist_t = validation(train_index, model, env, optimal_cuts)
random_cuts_t, random_hist_t = validation(train_index, rando, env, optimal_cuts)

# write hist to file and print results
with open(hist_filename, 'w') as f:
    f.write("Validation Results\n")
    f.write("Correct: %s\n" % hist["correct"])
    f.write("Incorrect: %s\n" % hist["incorrect"])
    f.write("Accuracy: %s\n" % (hist["correct"] / (hist["correct"] + hist["incorrect"])))
    f.write("Random Accuracy: %s\n" % (random_hist["correct"] / (random_hist["correct"] + random_hist["incorrect"])))

    f.write("\nTraining Results\n")
    f.write("Correct: %s\n" % hist_t["correct"])
    f.write("Incorrect: %s\n" % hist_t["incorrect"])
    f.write("Accuracy: %s\n" % (hist_t["correct"] / (hist_t["correct"] + hist_t["incorrect"])))
    f.write("Random Accuracy: %s\n" % (random_hist_t["correct"] / (random_hist_t["correct"] + random_hist_t["incorrect"])))

    print("Validation Results")
    print("Correct:", hist["correct"])
    print("Incorrect:", hist["incorrect"])
    print("Accuracy:", hist["correct"] / (hist["correct"] + hist["incorrect"]))
    print("Random Accuracy:", random_hist["correct"] / (random_hist["correct"] + random_hist["incorrect"]))

    print("\nTraining Results")
    print("Correct:", hist_t["correct"])
    print("Incorrect:", hist_t["incorrect"])
    print("Accuracy:", hist_t["correct"] / (hist_t["correct"] + hist_t["incorrect"]))
    print("Random Accuracy:", random_hist_t["correct"] / (random_hist_t["correct"] + random_hist_t["incorrect"]))

# # plot histogram
# plt.title("Validation Results")
# plt.xlabel("Correct/Incorrect")
# plt.ylabel("Number of Circuits")

# # show as percentages


# plt.legend()
# fig = plt.gcf()
# plt.show()
# fig.savefig(hist_filename)

######## Save Data ########
csv_filename = root_dir + date_str + "_" + str(max_run + 1) + "_data" + ".csv"

# put all data in one csv
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Actor Average", running_average])
    writer.writerow(["Random Average", random_average])
    writer.writerow(["Episode Rewards", "Random Rewards"])

    for i in range(len(episode_rewards)):
        writer.writerow([episode_rewards[i], random_rewards[i]])

######## Plot Data ########
plot_filename = root_dir + date_str + "_" + str(max_run + 1) + "_plot" + ".png"

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