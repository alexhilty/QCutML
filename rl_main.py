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
load_dataset = True # load dataset from file
dataset_filename = "../../qcircml_code/data_07312023/9/07312023_9_dataset.p" # filename of batched dataset
batch_size = 30
loops = 100
train_percent = 0.8

# model parameters
action_size = 6 # number of actions the agent can take
fc_layer_list = [1024, 256, 128] # list of number of hidden units for each desired fully connected layer

learning_rate = 0.01 # learning rate for optimizer
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
critic_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM) # define critic loss function
load = False # load model weights from file
model_load_filename = "../../qcircml_code/data_07282023_2/07282023_14_weights.h5" # filename of model weights

validate_with_best = True # validate with best checkpoint

# training parameters
window_size = 100 # size of window for moving average

# saving parameters
save = True # save data to file
root_dir = "../../qcircml_code/data_" + datetime.datetime.now().strftime("%m%d%Y") + "/"
date_str = datetime.datetime.now().strftime("%m%d%Y") # used for saving data

def model_save_condition(moving_averages, last_checkpoint, window_size): # function to determine when to save model
    if len(moving_averages) < window_size:
        return False
    
    if len(moving_averages) == window_size:
        # print("First checkpoint:", moving_averages[-1])
        return True
    
    if moving_averages[-1] > moving_averages[last_checkpoint] + 4:
        # print("Checkpoint:", moving_averages[-1])
        return True

# notes
notes = ""

######## Create Root Directory ########
# check if root_dir exists, if not create it
if not os.path.exists(root_dir):
    os.mkdir(root_dir)

# get list of all folders in root_dir
folders = [folder for folder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, folder))]

# make list of substring of all filenames after last underscore
runs = [int(foldername) for foldername in folders] # this is the run number
max_run = max(runs) if len(runs) > 0 else -1 # get max run number

root_dir = root_dir + str(max_run + 1) + "/" # create subfolder for this run

# check if root_dir exists, if not create it
if not os.path.exists(root_dir):
    os.mkdir(root_dir)

print("root_dir:", root_dir)

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
    "load_model": load,
    "model_load_filename": model_load_filename,
    "window_size": window_size,
    "save": save,
    "root_dir": root_dir,
    "date_str": date_str,
    "notes": notes,
    "validate_with_best": validate_with_best,
    "load_dataset": load_dataset,
    "dataset_filename": dataset_filename
}

######## Set Seed ########
np.random.seed(seed)
# tf.random.set_seed(seed)

# load circuit collection
circol = pickle.load(open(circ_filename, "rb"))

######## Create Batched Dataset ########
if load_dataset:
    train_data, train_index, val_data = pickle.load(open(dataset_filename, "rb"))
else:
    dataset_filename = root_dir + date_str + "_" + str(max_run + 1) + "_dataset"  + ".p"
    train_data, train_index, val_data = create_dataset(batch_size, loops, circol, train_percent)
    pickle.dump((train_data, train_index, val_data), open(dataset_filename, "wb"))

# print("train_data:", train_data.shape)
# print("val_data:", val_data.shape)

######## Create Environment and Model ########

env = CutEnvironment(circol) # create cut environment
model = CutActorCritic(action_size, fc_layer_list) # create model
rando = RandomSelector(action_size)

if load:
    image_shape = (circol.images[-1][0].shape[0], circol.images[-1][0].shape[1])
    dummy = tf.zeros((1, image_shape[0], image_shape[1]))

    # print("image_shape:", image_shape)
    # print("dummy:", str(dummy))
    
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
episode_rewards, random_rewards, running_average, random_average = train_loop(train_data, model, rando, env, critic_loss, optimizer, model_save_condition, window_size, model_save_filename)

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

######## Validate Model ########
print("Model Calls:", model.call_count)
if validate_with_best:
    # find filename of latest checkpoint
    checkpoints = [filename for filename in os.listdir(root_dir) if (filename.endswith(".h5") and not filename.endswith("final.h5"))]
    checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    model_load_filename = root_dir + checkpoints[-1]

    # load model weights
    new_model = CutActorCritic(action_size, fc_layer_list)
    image_shape = (circol.images[-1][0].shape[0], circol.images[-1][0].shape[1])
    dummy = tf.zeros((1, image_shape[0], image_shape[1]))

    # print("image_shape:", image_shape)
    # print("dummy:", str(dummy))
    
    action_logits_c, values = new_model(dummy) # call model once to initialize weights
    new_model.load_weights(model_load_filename)
    print("Loaded Model Weights:", model_load_filename)

    model = new_model

print("Model Calls:", model.call_count)

hist_filename = root_dir + date_str + "_" + str(max_run + 1) + "_hist" + ".txt"

optimal_cuts, optimal_circuits_index = compute_best_cuts(circol)
chosen_cuts, hist = validation2(val_data, model, env, optimal_cuts)
random_cuts, random_hist = validation2(val_data, rando, env, optimal_cuts)

chosen_cuts_t, hist_t = validation2(train_index, model, env, optimal_cuts)
random_cuts_t, random_hist_t = validation2(train_index, rando, env, optimal_cuts)

# # write hist to file and print results
# with open(hist_filename, 'w') as f:

random_full = np.concatenate((random_hist, random_hist_t))
plot_hist = [hist, hist_t, random_full]
# plot multi-bar histogram using hist, random_hist, hist_t, random_hist_t in matplotlib
colors = ['red', 'tan', 'lime']
labels = ['Agent Validation Data', 'Agent Training Data', 'Random']
res = plt.hist(plot_hist, bins=range(-1, max(random_full) + 3), color=colors, label=labels, density=True, align='left')
plt.legend()
plt.title("Histogram of Gate Cut Depth Difference")
plt.xlabel("Gate Cut Depth Difference")
plt.ylabel("Percent")
# show all x ticks
plt.xticks(range(-1, max(random_full) + 3))

# show percentage above each bar in histogram
max = 0
for i in range(len(res[0])):
    for j in range(len(res[1]) - 1):
        if res[0][i][j] != 0:
            plt.text(res[1][j] + (-0.36 + 0.265 * i), res[0][i][j] + 0.01, str(round(res[0][i][j] * 100)) + "%", rotation=90)
        
        if res[0][i][j] > max:
            max = res[0][i][j]

# set y axis limit to max + text height
plt.ylim(0, max + 0.1)
fig = plt.gcf()
fig.savefig(root_dir + date_str + "_" + str(max_run + 1) + "_hist" + ".png")
plt.show()