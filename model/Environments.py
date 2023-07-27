import math as m

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
        self.state = None # current state of the environment (circuit index)

    # defining environment step (cutting a circuit)
    # the action is index of the gate to cut (column of image to remove)
    def cut(self, action):
        '''Defines the environment step (cutting a circuit)
        
        The action is index of the gate to cut (column of image to remove)
        
        Parameters
        ------------
            action: int
                index of the gate to cut (column of image to remove)

        Returns
        ------------
            reward: float
                reward for the action
            state_image: np.array
                image of the new state
        '''
        
        # check if action is valid
        if action < 0 or action >= len(self.circol.circuits[self.state[0]][self.state[1]]):
            raise Exception("Invalid action: " + str(action))

        # remove gate from circuit
        gates = list(self.circol.circuits[self.state[0]][self.state[1]])
        gates.pop(action)

        # get new state
        new_state = self.circol.gates_to_index(gates)

        # # compute reward (negative depth difference) (old - new)
        # NOTE: maybe later scale with max possible improvement of each circuit
        reward = self.circol.q_transpiled[self.state[0]][self.state[1]].depth() - self.circol.q_transpiled[new_state[0]][new_state[1]].depth() - 1
        reward = reward / abs(reward) * reward ** 2 if reward != 0 else 0

        # reward = self.circol.q_transpiled[self.state[0]][self.state[1]].depth() / self.circol.q_transpiled[new_state[0]][new_state[1]].depth()

        # set new state
        self.state = new_state

        # return reward, state image, done flag
        return reward, self.get_image()

    # set the current circuit to the given index
    def set_state(self, n1, n2):
        '''Sets the current circuit to the given index.
        
        Parameters
        ------------
            n1: int
                first index of the circuit in the circuit collection
            n2: int
                second index of the circuit in the circuit collection
        '''

        self.state = (n1, n2)

    # get image for current state
    def get_image(self):
        '''Gets the image for the current state.'''

        return self.circol.images[self.state[0]][self.state[1]]