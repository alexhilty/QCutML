from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

# defining agent/critic
class CutActorCritic(tf.keras.Model):
    '''Neural Network Model for Actor Critic Agent. Inherits from tf.keras.Model.'''

    def __init__(self, num_actions: int, fc_layer_list: list):
        '''Initialize the Actor and Critic Model
        
        Defines the model layers using fc_layer_list (all fully connected layers)

        Parameters
        ------------
            num_actions: int
                number of actions the agent can take
            fc_layer_list: list<int>
                list of number of hidden units for each desired fully connected layer
        '''
        super().__init__()

        self.flat = layers.Flatten() # start by flattening image
        self.call_count = 0

        # NOTE: for now use fully connected layers (like in tensorflow tutorial)
        self.common_layers = []

        for num_hidden_units in fc_layer_list:
            self.common_layers.append(layers.Dense(num_hidden_units, activation="relu"))

        # NOTE: later maybe completely separate actor and critic networks
        self.actor = layers.Dense(num_actions) # NOTE: actor returns a probability distribution over actions)
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor):
        '''Forward pass of the model

        Implements forward pass using model layers and inputs

        Parameters
        ------------
            inputs: tf.Tensor
                input tensor to the model
        '''

        self.call_count += 1

        x = self.flat(inputs) # NOTE: flatten image for now (maybe later replace with convolutional layer)

        for layer in self.common_layers:
            x = layer(x)

        return self.actor(x), self.critic(x)
    
class RandomSelector(tf.keras.Model):

    def __init__(self, num_actions: int):
        super().__init__()

        self.num_actions = num_actions
        self.call_count = 0

    def call(self, inputs: tf.Tensor):

        # print(tf.random.uniform(shape=(inputs.shape[0], self.num_actions)))

        # create a tensor with number 1 in a random position for each batch
        # print(inputs.shape)

        self.call_count += 1

        uni = tf.repeat([1/inputs.shape[2]], inputs.shape[2])
        uni = tf.repeat([uni], repeats = [inputs.shape[0]], axis = 0)

        # print(uni)

        # quit()
        return uni, tf.ones(shape=(inputs.shape[0], 1))
    
# an actor that will always pick the same one
class FixedSelector(tf.keras.Model):

    def __init__(self, num_actions:int, constant_action:int):
        super().__init__()

        self.num_actions = num_actions
        self.constant_action = constant_action
        self.call_count = 0

    def call(self, inputs: tf.Tensor):
        # create a tensor of length num_actions with all zeros except for a 1 in the constant_action position
        self.call_count += 1

        uni = np.zeros(self.num_actions)
        uni[self.constant_action] = 1
        uni = tf.convert_to_tensor(uni, dtype=tf.float32)
        uni = tf.repeat([uni], repeats = [inputs.shape[0]], axis = 0)

        return uni, tf.ones(shape=(inputs.shape[0], 1))