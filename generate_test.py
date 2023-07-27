from CircuitCollection import CircuitCollection as cc
import itertools

from qiskit import *
from qiskit.transpiler import CouplingMap
from qiskit.tools.jupyter import *
from qiskit.visualization import *

import time
import pickle

def main():
    start_time = time.time()

    # Initialize a 4 qubit circuit
    n = 4
    trials = 10
    gates = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)]

    # circuit collection
    circol = cc(gates, n, 2)

    # generate all children
    circol.generate_circuits()

    # build all circuits
    circol.build_circuits()

    # transpile all circuits
    circol.transpile_circuits(n = 1, trials = trials, coupling_map = CouplingMap.from_line(n), optimization_level=1)

    # print(circol.num_circuits())

    # print run time
    print("--- Total: %s seconds ---" % (time.time() - start_time))

    # print circuits per second
    print("--- %s circuits per second ---" % (circol.num_circuits() * trials/(time.time() - start_time)))

    pickle.dump(circol, open("./data/circol_test.p", "wb"))

    # n1, n2 = circol.gates_to_index([(2, 3), (0, 3), (1, 2), (0, 1)])
    # print(n1, n2)
    # print(circol.circuits[n1][n2])

def main2():
    circol = pickle.load(open("./data/circol_test.p", "rb"))

    n1, n2 = circol.gates_to_index([(0, 1), (2, 3), (0, 2), (0, 3), (1, 3)])

    print(circol.q_circuits[n1][n2].draw())

    circol.convert_to_images()

    print(circol.images[n1][n2])

    # print(circol.num_circuits())

    # child = circol.child_indecies(n1, n2)

    # print("Children:")
    # print(*map(lambda x: circol.circuits[x[0]][x[1]], child), sep = "\n")

if __name__ == "__main__":
    # main()

    main2()