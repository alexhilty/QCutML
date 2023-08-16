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
from model.ActorCritic import Cutter, RandomSelector, CutterPointer
from model.Utils import *
from model.Environments import CutEnvironment

def run_model(
    ######## Parameters ########

    seed = 324, # seed for numpy and tensorflow

    circ_filename = "../../qcircml_code/data/circol_base_4qubits.p", # filename of circuit collection

    # batch parameters
    load_dataset = False, # load dataset from file
    dataset_filename = "../../qcircml_code/data_07312023/9/07312023_9_dataset.p", # filename of batched dataset
    batch_size = 30,
    loops = 300,
    train_percent = 0.8,

    # model parameters
    action_size = 6, # number of actions the agent can take
    model_type = ["rl", "attention"], # list of model types, must be same size as layer_lists (rl is normal, attention is pointer network)
    layer_lists = [[('flatten', None), ('fc', 1024), ('fc', 256), ('fc', 128)], [24, 100, [('fc', 256), ('fc', 128), ('fc', 64)]]], # list of lists of number of hidden units for each desired fully connected layer (one list for each model)
    # for layer_lists, if model_type is rl, then each element is a tuple of (layer_type, layer_size)
    # if model_type is attention, then the list is: [lstm_size, attention_size, g_model_list = [(layer_type, layer_size), ...]

    learning_rate = 0.01, # learning rate for optimizer# define critic loss function
    load = False, # load model weights from file
    model_load_filenames = ["../../qcircml_code/data_07282023_2/07282023_14_weights.h5"], # filename of model weights, must be same size as layer_lists
    transpose = [False, False], # whether to transpose the image before feeding it into the model, must be same size as layer_lists (no affect on attention model)

    validate_with_best = False, # validate with best checkpoint, FIXME: functionality broken with multiple models

    # training parameters
    window_size = 100, # size of window for moving average
    tf_function = True, # whether to use tf.function to speed up training

    # saving parameters
    save = True, # save data to file
    root_dir = "../../qcircml_code/data_" + datetime.datetime.now().strftime("%m%d%Y") + "/",
    date_str = datetime.datetime.now().strftime("%m%d%Y"), # used for saving data

    # notes
    notes = "",
    show_plot = True):

    # tf.config.run_functions_eagerly(True)

    ######## More Model Parameters ########

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    critic_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

    def model_save_condition(moving_averages, last_checkpoint, window_size): # function to determine when to save model
        # print(len(moving_averages), last_checkpoint, window_size)
        if len(moving_averages) < window_size:
            return False
        
        if len(moving_averages) == window_size:
            # print("First checkpoint:", moving_averages)
            return True
        
        if moving_averages[-1] > moving_averages[last_checkpoint] + 4:
            # print("Checkpoint:", moving_averages[-1])
            return True

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
        "model_type": model_type,
        "layer_list": layer_lists,
        "transpose": transpose,
        "learning_rate": learning_rate,
        "optimizer": type(optimizer),
        "critic_loss": type(critic_loss),
        "load_model": load,
        "model_load_filename": model_load_filenames,
        "window_size": window_size,
        "tf_function": tf_function,
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
        train_data, train_index, val_data = create_dataset(batch_size, loops, circol, train_percent, circol.depth)
        pickle.dump((train_data, train_index, val_data), open(dataset_filename, "wb"))

    # print("train_data:", train_data.shape)
    # print("val_data:", val_data.shape)

    ######## Create Environment and Model ########

    env = CutEnvironment(circol) # create cut environment
    models = []

    if len(transpose) != len(layer_lists) and transpose[0] == False: # default transpose to false
        transpose = [transpose[0]] * len(layer_lists)
    elif len(transpose) != len(layer_lists):
        raise Exception("Transpose must be same size as layer_lists")

    for i in range(len(model_type)):
        if model_type[i] == "attention":
            model = CutterPointer(layer_lists[i][0], layer_lists[i][1], layer_lists[i][2])
            models.append(model)
        elif model_type[i] == "rl":
            model = Cutter(action_size, layer_lists[i], transpose[i])
            models.append(model)
        else:
            raise Exception("Invalid model type:", model_type[i])

    rando = RandomSelector(action_size)
    models.append(rando)

    if load:
        image_shape = (circol.images[-1][0].shape[0], circol.images[-1][0].shape[1])
        dummy = tf.zeros((1, image_shape[0], image_shape[1]))

        # print("image_shape:", image_shape)
        # print("dummy:", str(dummy))
        
        for i in range(len(models)):
            action_logits_c, values = models[i](dummy)
            model[i].load_weights(model_load_filenames[i])

    ######## Save Parameters ########
    parameters_filename = root_dir + date_str + "_" + str(max_run + 1) + "_parameters"  + ".txt"

    # save parameters to file
    with open(parameters_filename, 'w') as f:
        for key in parameters.keys():
            f.write("%s: %s\n" % (key, parameters[key]))

    # pickle parameter dictionary
    parameters_pkl_filename = root_dir + date_str + "_" + str(max_run + 1) + "_parameters"  + ".p"
    pickle.dump(parameters, open(parameters_pkl_filename, "wb"))

    ######## Train Models ########
    if not os.path.exists(root_dir + "checkpoints/"):
        os.mkdir(root_dir + "checkpoints/")

    model_save_filename = root_dir + "checkpoints/" + date_str + "_" + str(max_run + 1) + "_weights" + ".h5"
    rewards, averages, moving_averages = train_loop(train_data, models, env, critic_loss, optimizer, model_save_condition, window_size, model_save_filename, tf_function)

    ######## Save Data ########
    csv_filename = root_dir + date_str + "_" + str(max_run + 1) + "_data" + ".csv"

    # put all data in one csv
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # save model types
        for j in range(len(models)):
            writer.writerow(["Model " + str(j), str(models[j]), "Random" if j == len(layer_lists) else str(layer_lists[j])])

        # save averages
        for j in range(len(models)):
            writer.writerow(["Average " + str(j), averages[j]])

        writer.writerow([])

        # save episode rewards
        writer.writerow(["Episode Rewards " + str(i) for i in range(len(models))])
        for i in range(len(rewards[0])):
            writer.writerow([rewards[j][i] for j in range(len(models))])

    ######## Plot Data ########
    # make a plot for each model, and one for all models (except random)
    for j in range(len(models) - 1):
        plot_filename = root_dir + date_str + "_" + str(max_run + 1) + "_plot_j" + str(j) + ".png"

        plt.cla()
        # plot episode rewards
        plt.title("Episode Rewards for Model " + str(j))
        plt.xlabel("Episode")
        plt.ylabel("Episode Reward")

        # plot moving average
        # moving_average = []
        # moving_avg_random = []
        # for i in range(len(rewards[j]) - 1):
        #     moving_average.append(statistics.mean(rewards[j][max(0, i - window_size):i + 1]))
        #     moving_avg_random.append(statistics.mean(rewards[j][max(0, i - window_size):i + 1]))

        plt.plot(moving_averages[j], color='k', label='Agent')
        plt.plot(moving_averages[-1], color='b', label='Random')
        # show horizontal line at random average
        plt.axhline(y=averages[-1], color='r', linestyle='-', label='Random Average')
        plt.legend()

        fig = plt.gcf()
        fig.savefig(plot_filename)
        if show_plot:
            plt.show()

    # plot all models together
    plot_filename = root_dir + date_str + "_" + str(max_run + 1) + "_plot_all" + ".png"

    plt.cla()
    # plot episode rewards
    plt.title("Episode Rewards for all Models")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")

    # plot moving average
    # moving_average = []
    # moving_avg_random = []
    # for i in range(len(rewards[j]) - 1):
    #     moving_average.append(statistics.mean(rewards[j][max(0, i - window_size):i + 1]))
    #     moving_avg_random.append(statistics.mean(rewards[j][max(0, i - window_size):i + 1]))

    for j in range(len(models) - 1):
        plt.plot(moving_averages[j], label='Agent ' + str(j))

    plt.plot(moving_averages[-1], color='b', label='Random')
    # show horizontal line at random average
    plt.axhline(y=averages[-1], color='r', linestyle='-', label='Random Average')
    plt.legend()

    fig = plt.gcf()
    fig.savefig(plot_filename)
    if show_plot:
        plt.show()

    ######## Validate Model ########
    # # print("Model Calls:", model.call_count)
    # if validate_with_best:
    #     # find filename of latest checkpoint
    #     checkpoints = [filename for filename in os.listdir(root_dir) if (filename.endswith(".h5") and not filename.endswith("final.h5"))]
    #     checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    #     model_load_filename = root_dir + checkpoints[-1]

    #     # load model weights
    #     new_model = Cutter(action_size, layers_)
    #     image_shape = (circol.images[-1][0].shape[0], circol.images[-1][0].shape[1])
    #     dummy = tf.zeros((1, image_shape[0], image_shape[1]))

    #     # print("image_shape:", image_shape)
    #     # print("dummy:", str(dummy))
        
    #     action_logits_c, values = new_model(dummy) # call model once to initialize weights
    #     new_model.load_weights(model_load_filename)
    #     print("Loaded Model Weights:", model_load_filename)

    #     model = new_model

    # print("Model Calls:", model.call_count)

    # hist_filename = root_dir + date_str + "_" + str(max_run + 1) + "_hist" + ".txt"

    # compute the best cuts
    optimal_cuts, optimal_circuits_index = compute_best_cuts(circol, depth=circol.depth)

    # validate all the models on validation and training data
    chosen_cut_list_v = []
    hist_list_v = []
    chosen_cut_list_t = []
    hist_list_t = []
    for i in range(len(models)):
        chosen_cuts, hist = validation2(val_data, models[i], env, optimal_cuts)
        chosen_cuts_t, hist_t = validation2(train_index, models[i], env, optimal_cuts)

        chosen_cut_list_v.append(chosen_cuts)
        hist_list_v.append(hist)
        chosen_cut_list_t.append(chosen_cuts_t)
        hist_list_t.append(hist_t)

    # # write hist to file and print results

    random_full = np.concatenate((hist_list_v[-1], hist_list_t[-1]))
    # create one histogram for each model and one for all models
    for k in range(len(models) - 1):
        plt.cla()

        plot_hist = [hist_list_v[k], hist_list_t[k], random_full]
        # plot multi-bar histogram using hist, random_hist, hist_t, random_hist_t in matplotlib
        colors = ['red', 'tan', 'lime']
        labels = ['Agent ' + str(k) +  ' Validation Data', 'Agent ' + str(k) + ' Training Data', 'Random']
        res = plt.hist(plot_hist, bins=range(-1, max(random_full) + 3), color=colors, label=labels, density=True, align='left')
        plt.legend()
        plt.title("Histogram of Gate Cut Depth Difference")
        plt.xlabel("Gate Cut Depth Difference")
        plt.ylabel("Percent")
        # show all x ticks
        plt.xticks(range(-1, max(random_full) + 3))

        # show percentage above each bar in histogram
        max_n = 0
        for i in range(len(res[0])):
            for j in range(len(res[1]) - 1):
                if res[0][i][j] != 0:
                    plt.text(res[1][j] + (-0.36 + 0.265 * i), res[0][i][j] + 0.012, str(round(res[0][i][j] * 100)) + "%", rotation=90)
                
                if res[0][i][j] > max_n:
                    max_n = res[0][i][j]

        # set y axis limit to max + text height
        plt.ylim(0, max_n + 0.1)
        fig = plt.gcf()
        fig.savefig(root_dir + date_str + "_" + str(max_run + 1) + "_hist_j" + str(k) + ".png")
        if show_plot:
            plt.show()

    # plot all models together
    plt.cla()

    plot_hist = [np.concatenate((hist_list_v[j], hist_list_t[j])) for j in range(len(models) - 1)] + [random_full]

    # colors = ['red', 'tan', 'lime']
    labels = ['Agent ' + str(j) for j in range(len(models) - 1)] + ['Random']
    res = plt.hist(plot_hist, bins=range(-1, max(random_full) + 3), label=labels, density=True, align='left')
    plt.legend()
    plt.title("Histogram of Gate Cut Depth Difference")
    plt.xlabel("Gate Cut Depth Difference")
    plt.ylabel("Percent")
    # show all x ticks
    plt.xticks(range(-1, max(random_full) + 3))

    # show percentage above each bar in histogram
    max_n = 0
    for i in range(len(res[0])):
        for j in range(len(res[1]) - 1):
            if res[0][i][j] != 0:
                plt.text(res[1][j] + (-0.33 + 0.45 * i), res[0][i][j] + 0.03, str(round(res[0][i][j] * 100)) + "%", rotation=90)
            
            if res[0][i][j] > max_n:
                max_n = res[0][i][j]

    # set y axis limit to max + text height
    plt.ylim(0, max_n + 0.1)
    fig = plt.gcf()
    fig.savefig(root_dir + date_str + "_" + str(max_run + 1) + "_hist_all" + ".png")
    if show_plot:
        plt.show()