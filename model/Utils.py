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
    # rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    # action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    # values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    # compute all images in batch
    images = env.convert_to_images_c(circuit_batch)

    action_logits_c, values = tf.map_fn(model, tf.expand_dims(images, 1), fn_output_signature=(tf.float32, tf.float32))
    # print(values)
    values = tf.squeeze(values, 1)
    
    # print(action_logits_c)
    # print(tf.squeeze(values))

    # print(action_logits_c, tf.squeeze(values, 1))

    action = tf.random.categorical(action_logits_c, 1)
    action_probs_c = tf.nn.softmax(action_logits_c) # compute log probability of actions
    action_probs = tf.gather_nd(action_probs_c, action, batch_dims = 1) # write chosen action probability to tensor array

    # reward, depths = env.cut([circuit_batch[i]], action)
    # y = lambda x: print(x[0], x[1])

    circuit_batch = tf.expand_dims(circuit_batch, 1)
    action = tf.expand_dims(action, 1)

    rewards, depths = tf.map_fn(env.cut, elems=[circuit_batch, action], fn_output_signature=(tf.float32, tf.int32))

    # print(rewards, depths)
    # quit()

    # iterate over batch of circuits
    # for i in tf.range(circuit_batch.shape[0]):

    #     # run model on images
    #     # action_logits_c, value = model(tf.expand_dims(images[i], 0))
    #     # values = values.write(i, tf.squeeze(value))

    #     # add batch dimension to action_logits_c and values
    #     # action_logits_c = tf.expand_dims(action_logits_c, 0)
    #     # values = tf.expand_dims(values, 0)

    #     # sample next action from the action probability distribution
    #     # action = tf.random.categorical(action_logits_c, 1)
    #     # action_probs_c = tf.nn.softmax(action_logits_c) # compute log probability of actions
    #     # action_prob = tf.gather_nd(action_probs_c, action, batch_dims = 1) # write chosen action probability to tensor array

    #     # action_probs = action_probs.write(i, tf.squeeze(action_prob))

    #     # apply action to environment to get next state and reward
    #     # FIXME: later get images here too
    #     reward, depths = env.cut([circuit_batch[i]], action)

    #     rewards = rewards.write(i, tf.squeeze(reward))

    # stack
    # action_probs = action_probs.stack()
    # values = values.stack()
    # rewards = rewards.stack()

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

        