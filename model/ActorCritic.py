from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

# defining agent/critic
class Cutter(tf.keras.Model):
    '''Neural Network Model for Actor Critic Agent. Inherits from tf.keras.Model.'''

    def __init__(self, num_actions: int, layers_def: list = [('fc', 256), ('fc', 128), ('fc', 64)], transpose: bool = False):
        '''Initialize the Actor and Critic Model
        
        Defines the model layers using fc_layer_list (all fully connected layers)
        The last layer of the actor and critic are pre-defined fully connected layers

        Parameters
        ------------
            num_actions: int
                number of actions the agent can take
            layers: list
                list of tuples that define the layers (in order) of the model
                fc -> fully connected -> layer size
                conv -> convolutional -> window size
                lstm -> long short term memory -> number of units
                flatten -> flatten image -> no parameters
            transpose: bool
                whether to transpose the image before feeding it into the model
        '''
        super().__init__()

        self.layers_list = []
        self.call_count = 0
        self.num_actions = num_actions
        self.transpose = transpose
        self.layers_def = layers_def

        # Define layers
        for layer in layers_def:
            if layer[0] == 'fc':
                self.layers_list.append(layers.Dense(layer[1], activation='tanh'))
            elif layer[0] == 'conv':
                self.layers_list.append(layers.Conv2D(filters = 1, kernel_size = (layer[1][0], layer[1][1]), activation='relu', padding='valid')) #FIXME: add input_shape here?
            elif layer[0] == 'lstm':
                self.layers_list.append(layers.LSTM(layer[1], activation = 'relu'))
            elif layer[0] == 'flatten':
                self.layers_list.append(layers.Flatten())

        # NOTE: later maybe completely separate actor and critic networks
        self.actor = layers.Dense(num_actions) # NOTE: actor returns a probability distribution over actions)
        self.critic = layers.Dense(1)

    # @tf.function
    def call(self, inputs: tf.Tensor):
        '''Forward pass of the model

        Implements forward pass using model layers and inputs

        Parameters
        ------------
            inputs: tf.Tensor
                input tensor to the model
        '''

        self.call_count += 1

        if self.transpose:
            x = tf.transpose(inputs, perm=[0, 2, 1]) # NOTE: later maybe add "gate_grouping" functionality
        else:
            x = inputs

        for i, layer in enumerate(self.layers_list):

            if self.layers_def[i][0] == 'conv':
                x = tf.expand_dims(x, axis=-1) # add channel dimension
                x = layer(x)
                x = tf.squeeze(x, axis=-1) # remove channel dimension
            else:
                x = layer(x)

        # print(x.shape)

        return self.actor(x), self.critic(x)
    
class RandomSelector(tf.keras.Model):

    def __init__(self, num_actions: int):
        super().__init__()

        self.num_actions = num_actions
        self.call_count = 0

    def call(self, inputs: tf.Tensor):

        # create a tensor with number 1 in a random position for each batch

        self.call_count += 1

        uni = tf.repeat([1/inputs.shape[2]], inputs.shape[2])
        uni = tf.repeat([uni], repeats = [inputs.shape[0]], axis = 0)

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