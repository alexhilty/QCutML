import math as m
import tensorflow as tf
import numpy as np

class CutEnvironment:
    '''Defines the environment for the cutting problem that the agen will interact with.'''

    def __init__(self, circuit_dataset = None):
        '''Initializes the environment
        
        Parameters
        ------------
            circuit_collection: CircuitCollection
                collection of circuits to use for the environment
        '''

        self.circol = circuit_dataset

        self.image_shape = (self.circol.num_qubits, sum(self.circol.reps)) # FIXME: allow for variable image size

    def cut_numpy_single(self, circuit_batch: np.array, actions: np.array):
        state = circuit_batch[0]
        action = actions[0][0]

        # print("\nCut Numpy")
        # print(state)
        # print(action)

        # remove gate from circuit
        gates = list(self.circol.current_section.circuits[state[0]][state[1]])

        if action < len(gates):
            gates.pop(action)

        # get new state
        new_state = self.circol.gates_to_index(gates)
        # print(self.circol.current_section.images[new_state[0]][new_state[1]].numpy())

        # # compute reward (negative depth difference) (old - new)

        reward = self.circol.current_section.q_transpiled[state[0]][state[1]].depth() - self.circol.current_section.q_transpiled[new_state[0]][new_state[1]].depth() - 1
        reward = reward / abs(reward) * reward ** 2 if reward != 0 else 0

        depth = self.circol.current_section.q_transpiled[new_state[0]][new_state[1]].depth()

        return np.array([reward], dtype=np.float32), np.array([depth], dtype =np.int32)


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
            gates = list(self.circol.current_section.circuits[state[0]][state[1]])

            if action < len(gates):
                gates.pop(action)

            # get new state
            new_state = self.circol.gates_to_index(gates)

            # # compute reward (negative depth difference) (old - new)
            # NOTE: maybe later scale with max possible improvement of each circuit
            reward = self.circol.current_section.q_transpiled[state[0]][state[1]].depth() - self.circol.current_section.q_transpiled[new_state[0]][new_state[1]].depth() - 1
            reward = reward / abs(reward) * reward ** 2 if reward != 0 else 0

            rewards.write(i, reward).mark_used()
            depths.write(i, self.circol.current_section.q_transpiled[new_state[0]][new_state[1]].depth()).mark_used()

            # reward = self.circol.q_transpiled[self.state[0]][self.state[1]].depth() / self.circol.q_transpiled[new_state[0]][new_state[1]].depth()
            
        rewards = rewards.stack().numpy()
        depths = depths.stack().numpy()
        return rewards, depths#, self.get_image()
    
    # wrapper for use as tensorflow function
    def cut(self, circ_action: tuple):
        '''See cut_numpy() for details.'''

        # return tuple(tf.numpy_function(self.cut_numpy, [circuit_batch, actions], (tf.float32, tf.int32))) # FIXME: numpy_function has some limitations

        circuit_batch = circ_action[0]
        actions = circ_action[1]
        return tuple(tf.numpy_function(self.cut_numpy_single, [circuit_batch, actions], (tf.float32, tf.int32))) # FIXME: numpy_function has some limitations
    
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

        # get images
        images = tf.gather_nd(self.circol.current_section.images, indexes)

        # images = images.to_tensor(shape = (index_shape[0], self.image_shape[0], self.image_shape[1]))

        return images