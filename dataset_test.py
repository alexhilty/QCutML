from CircuitCollection import CircuitDataset as cd
import pickle
from model.Environments import CutEnvironment
import model.Utils as utils
import model.ActorCritic as models

from qiskit import *
from qiskit.transpiler import CouplingMap
from qiskit.tools.jupyter import *
from qiskit.visualization import *

import time
import pickle
import numpy as np
import tensorflow as tf

root_dir = "../../qcircml_code/data/circset_4qubits_7gates_depth3"


def main():
    # Initialize a 4 qubit circuit
    n = 4
    trials = 10
    gates = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)]
    # gates = [(0, 1), (1, 2), (2, 3)]
    # reps = list(np.ones(len(gates)))
    reps = [2, 1, 1, 1, 1, 1]

    # circuit collection
    circol = cd(gates, n, 3, reps, 2500, root_dir)

    # generate all children
    circol.generate_circuits()

    # build all circuits
    circol.build_circuits()

    # convert to images
    circol.convert_to_images()

    # transpile all circuits
    circol.transpile_circuits(n = 1, trials = trials, coupling_map = CouplingMap.from_line(n), optimization_level=1)

    # set train percent
    circol.set_batches(0.8, 90, 1000)

    # compute optimal depths
    circol.compute_optimal_depths()

    # pickle dataset
    pickle.dump(circol, open(root_dir + "/dataset.p", "wb"))

def main2():
    # load dataset
    circol = pickle.load(open(root_dir + "/dataset.p", "rb"))

    circol.set_batches(0.7, 90, 50)
    circol.compute_optimal_depths()

    # dump dataset
    pickle.dump(circol, open(root_dir + "/dataset.p", "wb"))

    # # circol.compute_optimal_depths()

    # for i in range(len(circol.pickle_list)):
    #     circol.load_section(i)

    # # dump dataset
    #     print(len(circol.current_section.best_depths))

    # circol.set_batch_size(10)

    # for i, batch in enumerate(circol):
    #     print(f'{i}: {batch}')

    # print('')

    # circol.set_train(False)
    # circol.reset()

    # for i, batch in enumerate(circol):
    #     print(f'{i}: {batch}')


if __name__ == "__main__":
    # main()

    main2()