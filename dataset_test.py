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

root_dir = "../../qcircml_code/data/circset_4qubits_9gates_depth4_onebatch"


def main():
    # Initialize a 4 qubit circuit
    n = 4
    gates = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)]
    # gates = [(0, 1), (1, 2), (2, 3)]
    # reps = list(np.ones(len(gates)))
    reps = [2, 2, 2, 1, 1, 1]

    # circuit collection
    circol = cd(gates, n, 4, reps, 50000, root_dir)

    # generate all children
    circol.generate_circuits()

    # build all circuits
    circol.build_circuits()

    # convert to images
    circol.convert_to_images()

    # pickle dataset
    pickle.dump(circol, open(root_dir + "/dataset.p", "wb"))

    # transpile all circuits
    # circol.transpile_circuits(n = 10, trials = trials, coupling_map = CouplingMap.from_line(n), optimization_level=1)

    # set train percent
    # circol.set_batches(0.7, 90, 10)

    # compute optimal depths
    # circol.compute_optimal_depths()

    # print("\nNumber of Batches:", circol.batch_number)

    # pickle dataset
    # pickle.dump(circol, open(root_dir + "/dataset.p", "wb"))

def main2():
    # load dataset
    circol = pickle.load(open(root_dir + "/dataset.p", "rb"))

    # transpile some circuits
    trials = 10
    n = 4
    circol.transpile_circuits(n = 5, trials = trials, pickle_indecies = None, coupling_map = CouplingMap.from_line(n), optimization_level=1)

    circol.compute_optimal_depths()

    # set train percent
    # circol.set_batches(0.7, 90, 30)
    # count = 0
    # for circ in circol:
    #     count += 1

    # print(count)

    # for each pickle file print train batches
    # for file in circol.pickle_list:
    #     section = pickle.load(open(file, "rb"))
    #     print(len(section.train_batches))

    # compute optimal depths
    # circol.compute_optimal_depths()

    print("\nNumber of Batches:", circol.batch_number)

    # pickle dataset
    pickle.dump(circol, open(root_dir + "/dataset.p", "wb"))


if __name__ == "__main__":
    main()

    main2()