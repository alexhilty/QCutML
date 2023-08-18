import tensorflow as tf
import numpy as np
from model.Environments import CutEnvironment
from CircuitCollection import CircuitCollection
import copy
import tqdm
import statistics

################################################################################################
# FUNCTIONS FOR TRAINING
################################################################################################

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
    # print(images)

    # run model on images
    action_logits_c, values = model(images)
    values = tf.squeeze(values)

    # add batch dimension to action_logits_c and values
    action_logits_c = tf.expand_dims(action_logits_c, 0)
    values = tf.expand_dims(values, 0)

    # sample next action from the action probability distribution
    action = tf.random.categorical(action_logits_c, 1)
    action_probs_c = tf.nn.softmax(action_logits_c) # compute log probability of actions
    action_probs = tf.gather_nd(action_probs_c, action, batch_dims = 1) # write chosen action probability to tensor array

    # apply action to environment to get next state and reward
    # FIXME: later get images here too
    rewards, depths = env.cut(circuit_batch, action)

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

# define training step
# @tf.function # compiles function into tensorflow graph for faster execution # NOTE: sometimes doesn't train properly with this enabled
def train_step(circuit_batch, model: tf.keras.Model, cut_env, critic_loss_func, optimizer: tf.keras.optimizers.Optimizer, gamma: float) -> tf.Tensor:
    '''Runs a model training step'''

    # print("Tracing train_step")

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
    grads = tape.gradient(loss, model.trainable_variables)

    # apply the gradients to the model's parameters
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # episode_reward = tf.math.reduce_sum(np.array(rewards) * 30 / circuit_batch.shape[0])
    episode_reward = tf.math.reduce_sum(tf.math.scalar_mul(30 / circuit_batch.shape[0], rewards))

    return episode_reward

# define training loop
def train_loop(model_list, env, critic_loss, optimizer, save_condition_func, window_size = 100, model_save_filename = None, tf_function = True):
    '''Runs the training loop for the model.
    
    Parameters
    ------------
        save_condition_func: function
            function that takes the current moving average list and returns a boolean indicating whether to save the model'''

    rewards = []
    averages = []
    moving_averages = []
    save_checkpoints = []

    if tf_function:
        functions = []

    for j in range(len(model_list)):
        rewards.append([])
        averages.append(0)
        moving_averages.append([])
        save_checkpoints.append(0)

        # compile train_step function into tensorflow graph for faster execution
        if tf_function:
            functions.append(tf.function(train_step)) # make one for each model to avoid graph recompilation

    data_iter = iter(env.circol) # create iterator for dataset # WARNING: this may not update the object properly
    t = tqdm.trange(env.circol.batch_number - 1) # for showing progress bar
    for i in t:

        # get next batch of data
        train_data = tf.convert_to_tensor(next(data_iter))  # add batch dimension?

        # if list(model_list[0].variables) != []:
        #     print(model_list[0].variables)
        #     quit()
        for j in range(len(model_list)):
            # print('j ----------------------:', j)

            # run training step
            if tf_function:
                episode_reward = int(functions[j](train_data, model_list[j], env, critic_loss, optimizer, gamma=0.99))
            else:
                episode_reward = int(train_step(train_data, model_list[j], env, critic_loss, optimizer, gamma=0.99))

            # store episode reward
            rewards[j].append(episode_reward)

            # keep running average of episode rewards
            averages[j] = statistics.mean(rewards[j])

            # calculate average of last 100 episodes
            if i > window_size:
                moving_average = statistics.mean(rewards[j][i - 100:i])
            else:
                moving_average = averages[j]

            moving_averages[j].append(moving_average)

            if save_condition_func(moving_averages[j], save_checkpoints[j], window_size):
                model_list[j].save_weights(model_save_filename.split(".h5")[0] + "_j" + str(j) + "_i" + str(i) + ".h5")
                save_checkpoints[j] = i

        # update tqdm (progress bar)
        t.set_description("Checkpoint: {:05}, Running average: {:04.2f}, Moving average: {:04.2f}".format(save_checkpoints[0], averages[0], moving_averages[0][-1]))

    # save model
    for j in range(len(model_list)):
        model_list[j].save_weights(model_save_filename.split(".h5")[0] + "_j" + str(j) + "_final.h5")

    return rewards, averages, moving_averages

################################################################################################
# FUNCTIONS FOR DATASET CREATION
################################################################################################

# def create_dataset(batch_size: int, loops: int, circol: CircuitCollection, training_percent: float = 0.8, depth: int = 2):
#     '''Creates a dataset of circuits to train on'''
    
#     train_batch = []
#     # validation_batch = []
#     index_list_master = []

#     for j in range(depth - 1):
#         # index_list_master.append([[len(circol.circuits) - 1 - j, i] for i in range(0, len(circol.circuits[-1 - j])) ])  # create list of all possible circuit indexes
#         index_list_master += [[len(circol.circuits) - 1 - j, i] for i in range(0, len(circol.circuits[-1 - j]))]  # create list of all possible circuit indexes

#     # shuffle list
#     np.random.shuffle(index_list_master)

#     # put training_percent% of the data in the training set
#     train_index = index_list_master[:int(len(index_list_master) * training_percent)]
#     validation_index = index_list_master[int(len(index_list_master) * training_percent):]

#     for j in range(loops):

#         ###### train set
#         t = copy.deepcopy(train_index)
#         np.random.shuffle(t) # shuffle list
#         train_batch_temp = [t[i:i + batch_size] for i in range(0, len(t), batch_size)]

#         # remove last batch if it is not full
#         if len(train_batch_temp[-1]) != batch_size:
#             train_batch_temp.pop(-1)

#         train_batch.extend(train_batch_temp)

#     return tf.convert_to_tensor(train_batch, tf.int32), tf.convert_to_tensor(train_index, tf.int32), tf.convert_to_tensor(validation_index, tf.int32)

################################################################################################
# FUNCTIONS FOR VALIDATION
################################################################################################

# # function for computing best cuts on top level circuits
# def compute_best_cuts(circol: CircuitCollection, depth: int = 2):
#     optimal_circuits = []
#     optimal_cuts = []

#     for k in range (depth - 1):

#         optimal_cuts.insert(0, [])
#         optimal_circuits.insert(0, [])

#         for j in range(len(circol.circuits[-1 - k])): # loop through max lenght circuits

#             ind = circol.child_indecies(len(circol.circuits) - 1 - k, j) # compute children indecies
#             depths = [circol.q_transpiled[n1][n2].depth() for n1, n2 in ind]
#             min = depths[np.argmin(depths)]
#             min_indexes = np.where(np.array(depths) == min)[0]

#             optimal_circuits[0].append([ind[i] for i in min_indexes]) # choose child with lowest depth

#             # compute the index of the cut gate
#             parent_gates = circol.circuits[-1 - k][j]
#             child_gates = [circol.circuits[optimal_circuits[0][-1][i][0]][optimal_circuits[0][-1][i][1]] for i in range(len(optimal_circuits[0][-1]))]

#             temp = []
#             # for gate in parent_gates:
#             #     b = False

#             #     # check if gate is a best cut
#             #     for child_list in child_gates:
#             #         if gate not in child_list:
#             #             b = True
#             #             break

#             #     if b:
#             #         temp.append(parent_gates.index(gate))

#             for child_list in child_gates:
#                 for i in range(len(child_list)):
#                     if child_list[i] != parent_gates[i]:
#                         temp.append(i) ## FIXME: this works for now, but this is not guaranteed to be the exact cut index if multiple gates in a row (doens't matter now since we only care about the min depth at the validation step)
#                         break

#                     if i == len(child_list) - 1:
#                         temp.append(i + 1)

#             # print(temp)
#             # if temp == []:
#             #     print(ind)
#             #     print(parent_gates, child_gates)
#             #     quit()

                
#             optimal_cuts[0].append(temp)


#     return optimal_cuts, optimal_circuits

# another version of validation that gets more nuanced histogram data
def validation2(model, env, train_data = False):
    hist = {}
    difference = []

    # loop through sections
    for i in range(len(env.circol.pickle_list)):

        # load section
        env.circol.load_section(i)

        # get val indexes
        val_indexes = env.circol.current_section.val_indecies if not train_data else env.circol.current_section.train_indecies

        # get optimal depths
        optimal_depths = env.circol.current_section.best_depths

        # convert batch to images
        images = env.convert_to_images_c(tf.convert_to_tensor(val_indexes))

        # sample action from model
        action_logits_c, values = model(images)
        action = tf.random.categorical(action_logits_c, 1).numpy()

        # # choose first cut for each list in optimal_cuts
        # for i in range(len(val_indexes)):
        #     # print(optimal_cuts[val_indexes[i][0] - 1][val_indexes[i][1]])
        #     opt_cuts_temp.append(optimal_cuts[val_indexes[i][0] - 1][val_indexes[i][1]][0]) # NOTE: -1 because optimal_cuts is 1 shorter than val_indexes

        # opt_cuts_temp = tf.convert_to_tensor(opt_cuts_temp, tf.int32)
        # transpose
        # opt_cuts_temp = tf.expand_dims(opt_cuts_temp, 0)
        # opt_cuts_temp = tf.transpose(opt_cuts_temp)

        # cut circuits
        rewards, depths = env.cut(val_indexes, action) # cut with chosen cuts
        # rewards_opt, depths_opt = env.cut(val_indexes, opt_cuts_temp) # cut with optimal cuts

        # index optimal depths
        depths_opt = []
        for i in range(len(val_indexes)):
            depths_opt.append(optimal_depths[val_indexes[i][0]][val_indexes[i][1]])

        # compute difference in depth
        difference += list(depths.numpy() - np.array(depths_opt))

        # compute histogram
        for i in range(len(val_indexes)):
            if difference[i] not in hist.keys():
                hist[difference[i]] = 1
            else:
                hist[difference[i]] += 1
        
    return difference

        