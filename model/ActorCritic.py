from tensorflow.keras import layers
import tensorflow as tf

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

        x = self.flat(inputs) # NOTE: flatten image for now (maybe later replace with convolutional layer)

        for layer in self.common_layers:
            x = layer(x)

        return self.actor(x), self.critic(x)