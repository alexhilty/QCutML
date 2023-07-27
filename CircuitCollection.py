from qiskit import *
from qiskit.tools.jupyter import *
from qiskit.tools import parallel_map
from qiskit.visualization import *

import itertools
import math as m
import numpy as np

import operator as op
import time
import copy
    
# defines collections of all derivative circuits from seed circuit
class CircuitCollection:
    
    def __init__(self, seed = None, num_qubits = 0, depth = None):
        self.seed = seed # seed gate list
        self.num_qubits = num_qubits # number of qubits in circuit
        self.depth = depth if depth != None else len(seed)

        # class internal globals
        self.circuits = [] # list of child circuits
        self.q_circuits = [] # list of qiskit circuits
        self.q_transpiled = [] # list of transpiled qiskit circuits
        self.images = [] # list of image based representations of circuits

        self.generated_circuits = False
        self.built_circuits = False


    ####### UTILITIES #######
    # build circuit with qiskit
    def build_circuit(self, gates = []):

        qc = QuantumCircuit(self.num_qubits, 0)

        for gate in gates:
            qc.cx(gate[0], gate[1])

        return qc
    
    # Takes a set of gates and returns the corresponding index in circuits/q_circuits/q_transpiled
    def gates_to_index(self, gates = []):
        n1 = len(gates) - (len(self.seed) - self.depth) - 1
        n2 = 0

        seed = copy.deepcopy(self.seed)

        for i in range(len(gates)):
            
            # compute number of circuits that have this number of gates starting with a specific gate
            # print(str(len(seed) - 1) + "! / " + str(len(seed) - 1 - (n1 - 1 - i)) + "!")
            circ_num = m.factorial(len(seed) - 1) / m.factorial(len(seed) - (len(gates) - i))

            gate_index = seed.index(gates[i])

            # add to total index
            n2 += circ_num * gate_index

            # remove gate from seed
            seed.pop(gate_index)

        return (n1, int(n2))
    
    # returns the indecies of all direct children for a given index
    def child_indecies(self, n1, n2):
        if ( len(self.circuits[n1][n2]) - 1 < len(self.seed) - self.depth + 1):
            print("child_indecies: Circuit has no generated children!")
            return []

        children = itertools.combinations(self.circuits[n1][n2], len(self.circuits[n1][n2]) - 1)

        child_indecies = list(map(self.gates_to_index, children))

        return child_indecies
    
    def draw(self, n1, n2):
        return (self.q_circuits[n1][n2].draw(), self.q_transpiled[n1][n2].draw())
    
    # converts given circuit to image based representation
    # FIXME: fix argument structure
    def convert_to_image(self, *args):
        n1, n2 = args[0][0], args[0][1]
        circuit = self.circuits[n1][n2]

        image = np.zeros((len(self.seed), self.num_qubits)) # initialize image

        # convert each gate to a column of the image
        for i, gate in enumerate(circuit):
            # NOTE: set both to one (not preservering target/ctrl qubit information for now)
            image[i][gate[0]] = 1
            image[i][gate[1]] = 1

        # transpose image
        image = np.transpose(image)

        return image

    ####### USER FUNCTIONS #######

    def generate_circuits(self):
        gen_start_time = time.time()
        # generate all circuit lists
        self.circuits = []
        for i in range(len(self.seed) - self.depth + 1, len(self.seed) + 1):
            self.circuits.append(list(itertools.permutations(self.seed, i)))

        self.generated_circuits = True

        # print run time
        print("--- Generate: %s seconds ---" % (time.time() - gen_start_time))

    def build_circuits(self):
        build_start_time = time.time()

        if not self.generated_circuits:
            print("build_circuits: No Circuits Generated!")
            return
        
        # build and collect all circuits
        self.q_circuits = []
        for i in range(len(self.circuits)):
            self.q_circuits.append(list(map(self.build_circuit, self.circuits[i])))

        self.built_circuits = True

        # print run time
        print("--- Build: %s seconds ---" % (time.time() - build_start_time))

    # transpile circuits with n processes and transpile_args
    # trials specifies how many times to try transpile before choosing the best result (based on circuit depth)
    def transpile_circuits(self, n, trials, **transpile_args):
        trans_start_time = time.time()
        if not self.built_circuits:
            print("transpile_circuits: No Circuits Built!")
            return
        
        self.q_transpiled = []

        transpiled_temp = []

        # compute first set of transpiled circuits
        for i in range(len(self.q_circuits)):
            self.q_transpiled.append(
                parallel_map(transpile, self.q_circuits[i], num_processes = n, task_kwargs = transpile_args)
            )

        print("--- Transpile (Trial " + str(1) + "): %s seconds ---" % (time.time() - trans_start_time))


        # compute remaining trials and compare them to the first set, replacing them if they are better
        for i in range(trials - 1):
            for j in range(len(self.q_circuits)):

                # Transpile circuits
                transpiled_temp.append(
                    parallel_map(transpile, self.q_circuits[j], num_processes = n, task_kwargs = transpile_args)
                )

                # Compare circuits
                for k in range(len(self.q_circuits[j])):
                    if (transpiled_temp[j][k].depth() < self.q_transpiled[j][k].depth()):
                        self.q_transpiled[j][k] = transpiled_temp[j][k]

             # print run time
            print("--- Transpile (Trial " + str(i + 2) + "): %s seconds ---" % (time.time() - trans_start_time))
            transpiled_temp = []

        # print run time
        print("--- Transpile (" + str(trials) + "): %s seconds ---" % (time.time() - trans_start_time))

    # convert all circuits to image based representation
    def convert_to_images(self):
        # time how long it takes to convert all circuits
        convert_start_time = time.time()

        if not self.built_circuits:
            print("convert_to_images: No Circuits Built!")
            return

        for i in range(len(self.circuits)):
            # get index of all circuits
            indecies = map(self.gates_to_index, self.circuits[i])

            self.images.append(list(map(self.convert_to_image, indecies)))

        # print run time
        print("--- Convert: %s seconds ---" % (time.time() - convert_start_time))

    def num_circuits(self):
        sum = 0
        for i in range(len(self.circuits)):
            sum += len(self.circuits[i])
        
        return sum