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
        self.t_images = tf.convert_to_tensor(np.array(self.circol.images), dtype=tf.float32) # tensor of all images

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

        rewards = []
        for i in range(len(circuit_batch)):
            state = circuit_batch[i]
            action = actions[i][0]

            # remove gate from circuit
            gates = list(self.circol.circuits[state[0]][state[1]])
            gates.pop(action)

            # get new state
            new_state = self.circol.gates_to_index(gates)

            # # compute reward (negative depth difference) (old - new)
            # NOTE: maybe later scale with max possible improvement of each circuit
            reward = self.circol.q_transpiled[state[0]][state[1]].depth() - self.circol.q_transpiled[new_state[0]][new_state[1]].depth() - 1
            reward = reward / abs(reward) * reward ** 2 if reward != 0 else 0

            rewards.append(reward)

            # reward = self.circol.q_transpiled[self.state[0]][self.state[1]].depth() / self.circol.q_transpiled[new_state[0]][new_state[1]].depth()
            
        rewards = np.array(rewards)
        return np.ndarray.astype(rewards, np.float32) #, self.get_image()
    
    # wrapper for use as tensorflow function
    def cut(self, circuit_batch: tf.Tensor, actions: tf.Tensor):
        '''See cut_numpy() for details.'''

        return tf.numpy_function(self.cut_numpy, [circuit_batch, actions], [tf.float32]) # FIXME: numpy_function has some limitations

    # get image for current state
    def get_image(self, n: tf.Tensor = None):
        '''Gets the image for the current state.'''

        return tf.gather_nd(self.t_images, n)
    
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