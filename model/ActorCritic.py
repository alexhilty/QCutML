from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

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
        self.lstm = layers.LSTM(lstm_width, activation = 'relu', return_sequences = True)
        self.attention = Attention(attention_size)

    def call(self, inputs: tf.Tensor):
        '''Forward pass of the model'''

        # print(inputs)

        inputs = inputs.to_tensor(shape = (1, 4, 7)) # FIXME: allow for variable input size

        print(inputs.shape)

        # print(inputs)
        x = tf.transpose(inputs, perm=[0, 2, 1]) # transpose image for lstm
        # x = tf.map_fn(lambda y: tf.convert_to_tensor(np.transpose(y.numpy())), elems=inputs)

        print(x)

        x = self.lstm(x) # lstm layer, shape = (batch_size, num_gates, lstm_width)

        # compute g
        g = tf.zeros((x.shape[0], self.lstm_width)) # initialize g
        for layer in self.g_model_list:
            g = layer(g)
        g = self.out_g(g) # shape = (batch_size, lstm_width)
        
        # compute attention
        a = self.attention(x, g)

        return a, self.out_c(g) # size of a depends on number of gates in circuit

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
        inputs = inputs.to_tensor(shape = (1, 4, 7))

        self.call_count += 1

        uni = tf.repeat([1/inputs.shape[2]], inputs.shape[2])
        uni = tf.repeat([uni], repeats = [inputs.shape[0]], axis = 0)

        return tf.squeeze(uni), tf.squeeze(tf.ones(shape=(inputs.shape[0], 1)))