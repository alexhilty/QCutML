import tensorflow as tf
from model.Environments import CutEnvironment
import numpy as np

# function for running one episode
# one episode consists of performing one cut on a batch of circuits
# returns the total reward for the episode
def run_episode(circuit_batch, model, circuit_collection):
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

    # initialize environment
    env = CutEnvironment(circuit_collection)

    # compute all images in batch
    images = env.convert_to_images(circuit_batch)

    # run model on images
    action_logits_c, values = model(images)

    print("action_logits_c: " + str(action_logits_c))
    print("value: " + str(values))

    # sample next action from the action probability distribution
    action = tf.random.categorical(action_logits_c, 1)
    action_probs_c = tf.nn.softmax(action_logits_c) # compute log probability of actions
    action_probs = tf.expand_dims(tf.gather_nd(action_probs_c, action, batch_dims = 1), 1) # write chosen action probability to tensor array

    # apply action to environment to get next state and reward
    rewards = tf.expand_dims(env.cut(circuit_batch, action), 1)

    # print
    print("action: " + str(action))
    print("action_probs_c: " + str(action_probs_c))
    print("action_probs: " + str(action_probs))

    print("\nrewards: " + str(rewards))
    # print("images: " + str(images))



    # # FIXME: later modify to put all circuits in one batch and call model once
    # for i, circuit in enumerate(circuit_batch.numpy()):

    #     # set state
    #     env.set_state(circuit[0], circuit[1])

    #     # run the model and to get action probabilities and critic value
    #     image = tf.convert_to_tensor(env.get_image(), dtype=tf.float32)  # convert to tensor
    #     image = tf.expand_dims(image, 0)  # add batch dimension
    #     action_logits_c, value = model(image)

    #     # sample next action from the action probability distribution
    #     action = tf.random.categorical(action_logits_c, 1)[0, 0]
    #     action_probs_c = tf.nn.softmax(action_logits_c) # compute log probability of actions
    #     action_probs = action_probs.write(i, action_probs_c[0, action]) # write chosen action probability to tensor array

    #     # store critic values
    #     values = values.write(i, tf.squeeze(value))

    #     # apply action to the environment to get next state and reward
    #     action = np.array(action)
    #     # print("action: " + str(action))
    #     reward, image = env.cut(action)

    #     # print("reward: " + str(reward))
        
    #     # # nicely print action_logits_c, value, action
    #     # print("action_logits_c: " + str(action_logits_c))
    #     # print("value: " + str(value))
    #     # print("action: " + str(action))
    #     # print("\n")

    #     # store reward
    #     rewards = rewards.write(i, reward)

    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()

    return action_probs, values, rewards

# compute expected returns

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()
def get_expected_return(rewards: tf.Tensor, gamma: float, standardize: bool = True):
    '''Compute expected returns per step'''

    # NOTE: for now, just return rewards
    returns = rewards

    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) / 
               (tf.math.reduce_std(returns) + eps))


    return returns

# compute loss
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
def compute_loss(action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
    '''Computes the combined actor-critic loss'''

    # print(returns)
    # print(values)

    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

    critic_loss = huber_loss(values, returns)

    return actor_loss + critic_loss

# define training step
@tf.function
def train_step(circuit_batch, model: tf.keras.Model, circuit_collection, optimizer: tf.keras.optimizers.Optimizer, gamma: float) -> tf.Tensor:
    '''Runs a model training step'''

    with tf.GradientTape() as tape:

        # run the model for one episode to collect training data
        action_probs, values, rewards = run_episode(circuit_batch, model, circuit_collection)

        # calculate expected returns
        returns = get_expected_return(rewards, gamma)

        # convert training data to appropriate TF tensor shapes
        action_probs, values, returns = [
            tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

        # calculate loss values to update our network
        loss = compute_loss(action_probs, values, returns)

    # compute the gradients from the loss
    grads = tape.gradient(loss, model.trainable_variables)

    # apply the gradients to the model's parameters
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    episode_reward = tf.math.reduce_sum(rewards)

    return episode_reward