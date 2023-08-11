from model.ActorCritic import CutterPointer, Attention
import tensorflow as tf
from tensorflow.keras import layers
import time

# custom layer for attention
class AttentionTest(layers.Layer):

    def __init__(self, size: int = 100):
        super(AttentionTest, self).__init__()

        self.size = size

    def build(self, input_shape):
        self.WO = self.add_weight("kernel_WO", shape = (self.size, input_shape[-1]), initializer = "random_normal", trainable = True)
        self.WG = self.add_weight("kernel_WG", shape = (self.size, input_shape[-1]), initializer = "random_normal", trainable = True)
        self.v = self.add_weight("kernel_v", shape = (1, self.size), initializer = "random_normal", trainable = True)

    # take in two inputs and produce one output (first dimension of inputs is batch size)
    def call1(self, inputs: tf.Tensor, g: tf.Tensor):
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
    
    # take in two inputs and produce one output (first dimension of inputs is batch size)
    def call2(self, inputs: tf.Tensor, g: tf.Tensor):
        print(inputs.shape)
        print(self.WO.shape)
        wo_t = tf.repeat(self.WO, inputs.shape[2], axis = 0)
        print(wo_t.shape)


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

# testing attention layer
attT = AttentionTest(7)

batch_size = 5
num_gates = 10
lstm_width = 4

# call build to initialize weights
attT.build((batch_size, num_gates, lstm_width))

x = tf.random.uniform((batch_size, num_gates, lstm_width))
g = tf.random.uniform((batch_size, lstm_width))

# time each layer
start = time.time()
print(attT.call1(x, g))
print("Attention: --- %s seconds ---" % (time.time() - start))

start = time.time()
print("\n",attT.call2(x, g))
print("AttentionTest--- %s seconds ---" % (time.time() - start))