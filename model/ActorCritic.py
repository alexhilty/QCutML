from tensorflow.keras import layers
import tensorflow as tf

# defining agent/critic
class CutActorCritic(tf.keras.Model):

    def __init__(self, num_actions: int, num_hidden_units: int):
        '''Initialize'''
        super().__init__()
        self.flat = layers.Flatten()

        # NOTE: for now use fully connected layers (like in tensorflow tutorial)
        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.common2 = layers.Dense(num_hidden_units / 2, activation="relu")
        self.common3 = layers.Dense(num_hidden_units / 4, activation="relu")

        self.actor = layers.Dense(num_actions) # NOTE: actor returns a probability distribution over actions)
        self.critic = layers.Dense(1)

    def call(self, inputs):
        # print("Call: ", inputs.shape)

        fl = self.flat(inputs) # NOTE: flatten image for now (maybe later replace with convolutional layer)
        # print("fl: ", fl.shape)

        x = self.common(self.common2(self.common3(fl)))
        # print("x: ", x.shape)
        return self.actor(x), self.critic(x)