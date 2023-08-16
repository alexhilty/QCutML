import math as m
import tensorflow as tf
import numpy as np

class CutEnvironment:
    '''Defines the environment for the cutting problem that the agen will interact with.'''

    def __init__(self, circuit_collection = None):
        '''Initializes the environment
        
        Parameters
        ------------
            circuit_collection: CircuitCollection
                collection of circuits to use for the environment
        '''

        self.circol = circuit_collection

        # concatenate all lists of images into one numpy array
        self.n_images = np.array(self.circol.images[0])
        for i in range(1, len(self.circol.images)):
            self.n_images = np.concatenate((self.n_images, np.array(self.circol.images[i])))

        self.t_images = tf.convert_to_tensor(self.n_images, dtype = tf.float32) # tensor of all images
        self.t_images_breaks = tf.stack(np.cumsum(np.array([0] + [len(self.circol.images[i]) for i in range(len(self.circol.images))] ))) # tensor of all image breaks

    # defining environment step (cutting a circuit)
    # the action is index of the gate to cut (column of image to remove)
    def cut_numpy(self, circuit_batch: np.array, actions: np.array):
        '''Defines the environment step (cutting a circuit, wrapped by cut() for use as tensorflow function)
        
        The action is index of the gate to cut (column of image to remove)
        
        Parameters
        ------------
            actions: np.array
                indexes of the gate to cut (column of image to remove)
            circuit_batch: np.array
                batch of circuits to cut

        Returns
        ------------
            reward: float
                reward for the action
            state_image: np.array (FIXME: implement this)
                image of the new state
        '''

        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        depths = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        for i in range(len(circuit_batch)):
            state = circuit_batch[i]
            action = actions[i][0]

            # remove gate from circuit
            gates = list(self.circol.circuits[state[0]][state[1]])

            if action < len(gates):
                gates.pop(action)

            # get new state
            new_state = self.circol.gates_to_index(gates)

            # # compute reward (negative depth difference) (old - new)
            # NOTE: maybe later scale with max possible improvement of each circuit
            reward = self.circol.q_transpiled[state[0]][state[1]].depth() - self.circol.q_transpiled[new_state[0]][new_state[1]].depth() - 1
            reward = reward / abs(reward) * reward ** 2 if reward != 0 else 0

            rewards.write(i, reward).mark_used()
            depths.write(i, self.circol.q_transpiled[new_state[0]][new_state[1]].depth()).mark_used()

            # reward = self.circol.q_transpiled[self.state[0]][self.state[1]].depth() / self.circol.q_transpiled[new_state[0]][new_state[1]].depth()
            
        rewards = rewards.stack().numpy()
        depths = depths.stack().numpy()
        return rewards, depths#, self.get_image()
    
    # wrapper for use as tensorflow function
    def cut(self, circuit_batch: tf.Tensor, actions: tf.Tensor):
        '''See cut_numpy() for details.'''

        return tf.numpy_function(self.cut_numpy, [circuit_batch, actions], [tf.float32, tf.int32]) # FIXME: numpy_function has some limitations

    # get image for current state
    def get_image(self, n: tf.Tensor = None):
        '''Gets the image for the current state.'''

        n1, n2 = tf.split(n, num_or_size_splits=2)
        # tf.print(tf.cast(n1, tf.int32))
        # print(tf.get_static_value(n1), n2)

        image = tf.gather(self.t_images, tf.cast(tf.gather(self.t_images_breaks, n1), tf.int64) + tf.cast(n2, tf.int64))

        return tf.squeeze(image)
    
    def convert_to_images_c(self, indexes: tf.Tensor):
        '''Converts the circuits in the given batch to images.
        
        Parameters
        ------------
            indexes: tf.Tensor
                batch of indexes of the circuits to convert

        Returns
        ------------
            images: tf.Tensor
                batch of images of the converted circuits
        '''

        image_shape = self.circol.images[0][0].shape
        index_shape = indexes.shape

        # convert all circuits in batch to images using tf.scan
        images = tf.scan(
            lambda a, b: self.get_image(b), indexes, initializer = tf.zeros((image_shape[0], image_shape[1])))

        return images