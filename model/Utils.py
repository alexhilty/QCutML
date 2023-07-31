import tensorflow as tf
import numpy as np
from model.Environments import CutEnvironment
from CircuitCollection import CircuitCollection
import copy
import tqdm
import statistics

# function for running one episode
# one episode consists of performing one cut on a batch of circuits
# returns the total reward for the episode
def run_episode(circuit_batch, model, env: CutEnvironment):
    '''Runs a single episode to collect training data.
    
    Parameters
    ------------
        circuit_batch: tf.Tensor
            batch of circuits to run the episode on
        model: tf.keras.Model
            model to use for the episode
        circuit_collection: CircuitCollection
            collection of circuits to use for the episode
            
    Returns
    ------------
        action_probs: tf.Tensor
            action probabilities for the episode
        values: tf.Tensor
            critic values for the episode
        rewards: tf.Tensor
            rewards for the episode
        '''

    # initialize tensor arrays to store the observations
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    # compute all images in batch
    images = env.convert_to_images_c(circuit_batch)

    # print("images: " + str(images))

    # run model on images
    action_logits_c, values = model(images)
    values = tf.squeeze(values)

    # sample next action from the action probability distribution
    action = tf.random.categorical(action_logits_c, 1)
    action_probs_c = tf.nn.softmax(action_logits_c) # compute log probability of actions
    action_probs = tf.gather_nd(action_probs_c, action, batch_dims = 1) # write chosen action probability to tensor array

    # apply action to environment to get next state and reward
    # FIXME: later get images here too
    rewards, depths = env.cut(circuit_batch, action)

    # print
    # print("action_logits_c: " + str(action_logits_c))
    # print("action: " + str(action))
    # print("action_probs_c: " + str(action_probs_c))

    # print("\naction_probs: " + str(action_probs))
    # print("rewards: " + str(rewards))
    # print("values: " + str(values))
    # print("images: " + str(images))
    # print()

    return action_probs, values, rewards

# compute expected returns
def get_expected_return(rewards: tf.Tensor, gamma: float, standardize: bool = True):
    '''Compute expected returns per step'''

    # NOTE: for now, just return rewards
    returns = rewards

    # Small epsilon value for stabilizing division operations
    eps = np.finfo(np.float32).eps.item()

    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) / 
               (tf.math.reduce_std(returns) + eps))

    return returns

# compute loss
def compute_loss(action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor, critic_loss_func) -> tf.Tensor:
    '''Computes the combined actor-critic loss'''

    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

    critic_loss = critic_loss_func(values, returns)

    return actor_loss + critic_loss

def create_dataset(batch_size: int, loops: int, circol: CircuitCollection, training_percent: float = 0.8):
    '''Creates a dataset of circuits to train on'''
    
    train_batch = []
    # validation_batch = []
    index_list_master = [[len(circol.circuits) - 1, i] for i in range(0, len(circol.circuits[-1])) ] # create list of all possible circuit indexes

    # shuffle list
    np.random.shuffle(index_list_master)

    # put training_percent% of the data in the training set
    train_index = index_list_master[:int(len(index_list_master) * training_percent)]
    validation_index = index_list_master[int(len(index_list_master) * training_percent):]

    for j in range(loops):

        ###### train set
        t = copy.deepcopy(train_index)
        np.random.shuffle(t) # shuffle list
        train_batch_temp = [t[i:i + batch_size] for i in range(0, len(t), batch_size)]

        # remove last batch if it is not full
        if len(train_batch_temp[-1]) != batch_size:
            train_batch_temp.pop(-1)

        train_batch.extend(train_batch_temp)

        # ###### validation set
        # v = copy.deepcopy(validation_index)
        # np.random.shuffle(v) # shuffle list
        # validation_batch_temp = [v[i:i + batch_size] for i in range(0, len(v), batch_size)]

        # # remove last batch if it is not full
        # if len(validation_batch_temp[-1]) != batch_size:
        #     validation_batch_temp.pop(-1)

        # validation_batch.extend(validation_batch_temp)

    return tf.convert_to_tensor(train_batch, tf.int32), tf.convert_to_tensor(train_index, tf.int32), tf.convert_to_tensor(validation_index, tf.int32)

# define training step
# @tf.function # compiles function into tensorflow graph for faster execution # FIXME: doesn't train with this enabled for some reason
def train_step(circuit_batch, model: tf.keras.Model, cut_env, critic_loss_func, optimizer: tf.keras.optimizers.Optimizer, gamma: float) -> tf.Tensor:
    '''Runs a model training step'''

    with tf.GradientTape() as tape:

        # run the model for one episode to collect training data
        action_probs, values, rewards = run_episode(circuit_batch, model, cut_env)

        # calculate expected returns
        returns = get_expected_return(rewards, gamma)

        # convert training data to appropriate TF tensor shapes
        action_probs, values, returns = [
            tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

        # calculate loss values to update our network
        loss = compute_loss(action_probs, values, returns, critic_loss_func)

    # compute the gradients from the loss
    # print(model.trainable_variables)
    grads = tape.gradient(loss, model.trainable_variables)

    # apply the gradients to the model's parameters
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # quit(0)

    episode_reward = tf.math.reduce_sum(np.array(rewards) * 30 / circuit_batch.shape[0])

    return episode_reward

# define training loop
def train_loop(train_data, model, rando, env, critic_loss, optimizer, save_condition_func, window_size = 100, model_save_filename = None):
    '''Runs the training loop for the model.
    
    Parameters
    ------------
        save_condition_func: function
            function that takes the current moving average list and returns a boolean indicating whether to save the model'''

    episode_rewards = []
    random_rewards = []

    moving_averages = []
    save_checkpoint = 0

    t = tqdm.trange(len(train_data)) # for showing progress bar
    for i in t:
        # run train step
        episode_reward = int(train_step(train_data[i], model, env, critic_loss, optimizer, gamma=0.99))
        random_reward = int(train_step(train_data[i], rando, env, critic_loss, optimizer, gamma=0.99))

        # store episode reward
        episode_rewards.append(episode_reward)
        random_rewards.append(random_reward)

        # keep running average of episode rewards
        running_average = statistics.mean(episode_rewards)
        random_average = statistics.mean(random_rewards)

        # calculate average of last 100 episodes
        if i > window_size:
            moving_average = statistics.mean(episode_rewards[i - 100:i])
            random_moving_average = statistics.mean(random_rewards[i - 100:i])
        else:
            moving_average = running_average
            random_moving_average = random_average

        moving_averages.append(moving_average)

        if save_condition_func(moving_averages, save_checkpoint, window_size):
            # print(model_save_filename.split(".h5")[0] + "_ " + str(i) + ".h5")
            model.save_weights(model_save_filename.split(".h5")[0] + "_" + str(i) + ".h5")
            save_checkpoint = i

        # update tqdm (progress bar)
        t.set_description("Checkpoint: {:05}, Running average: {:04.2f}, Moving average: {:04.2f}".format(save_checkpoint, running_average, moving_average))

    # save model
    model.save_weights(model_save_filename.split(".h5")[0] + "_final.h5")

    return episode_rewards, random_rewards, running_average, random_average

# function for computing best cuts on top level circuits
def compute_best_cuts(circol: CircuitCollection):
    optimal_circuits = []
    optimal_cuts = []

    for j in range(len(circol.circuits[-1])): # loop through max lenght circuits
        ind = circol.child_indecies(len(circol.circuits) - 1, j) # compute children indecies
        depths = [circol.q_transpiled[n1][n2].depth() for n1, n2 in ind]
        min = depths[np.argmin(depths)]
        min_indexes = np.where(np.array(depths) == min)[0]

        optimal_circuits.append([ind[i] for i in min_indexes]) # choose child with lowest depth

        # compute the index of the cut gate
        parent_gates = circol.circuits[-1][j]
        child_gates = [circol.circuits[optimal_circuits[-1][i][0]][optimal_circuits[-1][i][1]] for i in range(len(optimal_circuits[-1]))]

        temp = []
        for gate in parent_gates:
            b = False

            # check if gate is a best cut
            for child_list in child_gates:
                if gate not in child_list:
                    b = True
                    break

            if b:
                temp.append(parent_gates.index(gate))
            
        optimal_cuts.append(temp)


    return optimal_cuts, optimal_circuits

# # # define validation loop
# def validation(val_data, model, env, best_cuts):
#     '''best_cuts is a list of the indecies of the best cuts for every circuit'''
#     chosen_cuts = []
#     hist = {"correct": 0, "incorrect": 0}

#     # convert batch to images
#     images = env.convert_to_images_c(val_data)

#     # sample action from model
#     action_logits_c, values = model(images)
#     action = tf.random.categorical(action_logits_c, 1).numpy()

#     for j in range(len(action)):
#         # store chosen cut
#         chosen_cuts.append(action[j][0])

#         # compare with best cut
#         best = best_cuts[val_data[j][1]]

#         if chosen_cuts[-1] in best:
#             hist["correct"] += 1
#         else:
#             hist["incorrect"] += 1

#     return chosen_cuts, hist

# another version of validation that gets more nuanced histogram data
def validation2(val_indexes, model, env, optimal_cuts):
    hist = {}
    opt_cuts_temp = []

    # convert batch to images
    images = env.convert_to_images_c(val_indexes)

    # sample action from model
    action_logits_c, values = model(images)
    action = tf.random.categorical(action_logits_c, 1).numpy()

    # choose first cut for each list in optimal_cuts
    for i in range(len(val_indexes)):
        opt_cuts_temp.append(optimal_cuts[val_indexes[i][1]][0])

    # print("opt_cuts_temp:", opt_cuts_temp)
    opt_cuts_temp = tf.convert_to_tensor(opt_cuts_temp, tf.int32)
    # transpose
    opt_cuts_temp = tf.expand_dims(opt_cuts_temp, 0)
    opt_cuts_temp = tf.transpose(opt_cuts_temp)

    # print("opt_cuts_temp:", opt_cuts_temp)

    # cut circuits
    rewards, depths = env.cut(val_indexes, action) # cut with chosen cuts
    rewards_opt, depths_opt = env.cut(val_indexes, opt_cuts_temp) # cut with optimal cuts

    # compute difference in depth
    difference = depths.numpy() - depths_opt.numpy()

    # compute histogram
    for i in range(len(val_indexes)):
        if difference[i] not in hist.keys():
            hist[difference[i]] = 1
        else:
            hist[difference[i]] += 1
    
    action = action.squeeze()
    return action, difference

        