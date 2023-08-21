from tensorflow.keras import layers
from tensorflow.keras import regularizers
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
                self.layers_list.append(layers.Dense(layer[1], activation='relu'))
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

        return self.actor(x), self.critic(x)
    
# an actor critic model that has dynamic output size (based on pointer network)
class CutterPointer(tf.keras.Model):

    def __init__(self, lstm_width: int = 20, attention_size: int = 100, g_model_def: list =  [('fc', 256), ('fc', 128), ('fc', 64)]):
        super().__init__()

        self.lstm_width = lstm_width
        self.attention_size = attention_size
        self.g_model_def = g_model_def
        self.g_model_list = []

        # Define g model layers
        for la in g_model_def:
            if la[0] == 'fc':
                self.g_model_list.append(layers.Dense(la[1], activation='relu'))
            elif la[0] == 'conv':
                self.g_model_list.append(layers.Conv2D(filters = 1, kernel_size = (la[1][0], la[1][1]), activation='relu', padding='valid'))

        self.out_g = layers.Dense(lstm_width) # last layer of g model, ensures output size is correct
        self.out_c = layers.Dense(1) # crictic output

        # defining lstm and attention layers
        self.lstm = layers.LSTM(lstm_width, activation = 'relu', return_sequences = True, kernel_regularizer=regularizers.l2(0.001))
        self.attention = Attention(attention_size)

    def call(self, inputs: tf.Tensor):
        '''Forward pass of the model'''

        x = tf.transpose(inputs, perm=[0, 2, 1]) # transpose image for lstm

        x = self.lstm(x) # lstm layer, shape = (batch_size, num_gates, lstm_width)

        # get last output
        # g = tf.map_fn(lambda batch: batch[-1], x) # shape = (batch_size, lstm_width)

        # remove last output in each batch
        # x = tf.map_fn(lambda batch: batch[:-1], x) # shape = (batch_size, num_gates - 1, lstm_width)
        # tf.print(g)

        # compute g
        # g = tf.zeros((x.shape[0], self.lstm_width)) # initialize g
        # for layer in self.g_model_list:
        #     g = layer(g)
        # g = self.out_g(g) # shape = (batch_size, lstm_width)
        # tf.print(g)
        
        ze = tf.zeros((x.shape[0], self.lstm_width))
        # compute attention
        a = self.attention(x, ze)
        # tf.print(a)

        return a, self.out_c(ze) # size of a depends on number of gates in circuit

# custom layer for attention
class Attention(layers.Layer):

    def __init__(self, size: int = 100):
        super(Attention, self).__init__()

        self.size = size

    def build(self, input_shape):
        self.WO = self.add_weight("kernel_WO", shape = (self.size, input_shape[-1]), initializer = "random_normal", trainable = True)
        self.WG = self.add_weight("kernel_WG", shape = (self.size, input_shape[-1]), initializer = "random_normal", trainable = True)
        self.v = self.add_weight("kernel_v", shape = (1, self.size), initializer = "random_normal", trainable = True)

    # take in two inputs and produce one output (first dimension of inputs is batch size)
    def call(self, inputs: tf.Tensor, g: tf.Tensor):
        term1 = tf.map_fn(
                    lambda batch: tf.map_fn(lambda x: tf.matmul(self.WO, tf.expand_dims(x, -1)), batch),
                    inputs
                )
        
        term2 = tf.map_fn(
                    lambda x: tf.matmul(self.WG, tf.expand_dims(x, -1)), g
                )
        
        tan_sum = tf.nn.tanh(term1 + tf.repeat(tf.expand_dims(term2, 1), inputs.shape[1], axis = 1))

        final = tf.map_fn(
                    lambda batch: tf.map_fn(lambda x: tf.matmul(self.v, x), batch),
                    tan_sum
                )

        return tf.squeeze(final)
    
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