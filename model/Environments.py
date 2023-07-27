# defining environment

import math as m
class CutEnvironment:

    def __init__(self, circuit_collection = None):
        self.circol = circuit_collection

        self.state = None # current state of the environment (circuit index)
        self.episode = 0
        self.done = False

    # defining environment step (cutting a circuit)
    # the action is index of the gate to cut (column of image to remove)
    def cut(self, action):
        # check if action is valid
        if action < 0 or action >= len(self.circol.circuits[self.state[0]][self.state[1]]):
            raise Exception("Invalid action: " + str(action))

        # remove gate from circuit
        gates = list(self.circol.circuits[self.state[0]][self.state[1]])
        gates.pop(action)

        # get new state
        new_state = self.circol.gates_to_index(gates)

        # # compute reward (negative depth difference) (old - new)
        reward = self.circol.q_transpiled[self.state[0]][self.state[1]].depth() - self.circol.q_transpiled[new_state[0]][new_state[1]].depth() - 1
        reward = reward / abs(reward) * reward ** 2 if reward != 0 else 0

        # reward = self.circol.q_transpiled[self.state[0]][self.state[1]].depth() / self.circol.q_transpiled[new_state[0]][new_state[1]].depth()

        # set new state
        self.state = new_state

        # return reward, state image, done flag
        return reward, self.get_image(), self.done

    # set the current circuit to the given index
    def set_state(self, n1, n2):
        self.state = (n1, n2)

    # get image for current state
    def get_image(self):
        return self.circol.images[self.state[0]][self.state[1]]