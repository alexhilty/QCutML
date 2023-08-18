from CircuitCollection import CircuitCollection as cc
import itertools

from qiskit import *
from qiskit.transpiler import CouplingMap
from qiskit.tools.jupyter import *
from qiskit.visualization import *

import time
import pickle
import numpy as np
import tensorflow as tf

def main():
    start_time = time.time()

    # Initialize a 4 qubit circuit
    n = 4
    trials = 10
    gates = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)]
    # gates = [(0, 1), (1, 2), (2, 3)]
    # reps = list(np.ones(len(gates)))
    reps = [2, 2, 2, 2, 1, 1]

    # circuit collection
    circol = cc(gates, n, 3, reps)

    # generate all children
    circol.generate_circuits()

    # build all circuits
    circol.build_circuits()

    # transpile all circuits
    circol.transpile_circuits(n = 10, trials = trials, coupling_map = CouplingMap.from_line(n), optimization_level=1)

    # print(circol.num_circuits())

    # print run time
    print("--- Total: %s seconds ---" % (time.time() - start_time))

    # print circuits per second
    print("--- %s circuits per second ---" % (circol.num_circuits() * trials/(time.time() - start_time)))

    pickle.dump(circol, open("../../qcircml_code/data/circol_base_4qubits_10gates_depth3_dict.p", "wb"))

    # n1, n2 = circol.gates_to_index([(2, 3), (0, 3), (1, 2), (0, 1)])
    # print(n1, n2)
    # print(circol.circuits[n1][n2])

def main2():
    circol = pickle.load(open("../../qcircml_code/data/circol_base_4qubits_10gates_depth3_dict.p", "rb"))

    # print(len(circol.circuits[-1]))
    # for circuit in circol.circuits[-1]:
    #     print(circuit)

    search = [(1, 2), (2, 3), (1, 3), (0, 3), (0, 2), (0, 1)]
    print(search)

    n1, n2 = circol.gates_to_index(search)

    print(circol.circuits[n1][n2])
    print("Circuits Matched:", list(circol.circuits[n1][n2]) == search)

    print("\n", circol.q_circuits[n1][n2].draw())

    # print("\n", circol.q_circuits[0][0].draw())
    # print('\n', circol.q_circuits[0][1].draw())
    # print('\n', circol.q_circuits[0][2].draw())

    circol.convert_to_images()

    print("\n", circol.images[n1][n2])

    print("\n", circol.num_circuits())

    # check if all images are the same size
    print(all([circol.images[n1][n2].shape == circol.images[n1][n2].shape for n1 in range(len(circol.circuits)) for n2 in range(len(circol.circuits[n1]))]))

    # tf.convert_to_tensor(np.array(circol.images), dtype=tf.float32)

    pickle.dump(circol, open("../../qcircml_code/data/circol_base_4qubits_10gates_depth3_dict.p", "wb"))

    # child = circol.child_indecies(n1, n2)

    # print("Children:")
    # print(*map(lambda x: circol.circuits[x[0]][x[1]], child), sep = "\n")

if __name__ == "__main__":
    main()

    main2()