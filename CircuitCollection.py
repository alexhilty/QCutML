from qiskit import *
from qiskit.tools.jupyter import *
from qiskit.tools import parallel_map
from qiskit.visualization import *

import itertools
import math as m
import numpy as np
import tensorflow as tf

import operator as op
import time
import copy
import pickle

def unique_elem_it(iterator):
    '''Returns a list of unique elements from an iterator (preserving order)'''

    yield from dict.fromkeys(iterator)

##### CIRCUIT COLLECTION TEMPLATE #####
# defines the template for circuit collections
class CircuitCollectionTemplate:

    def __init__(self):
        self.q_transpiled = []
        self.q_circuits = []
        self.circuits = []
        self.images = []

        self.built_circuits = False
    
    # Takes a set of gates and returns the corresponding index in circuits/q_circuits/q_transpiled
    def gates_to_index(self, gates = []):
        raise NotImplementedError

    # returns the indecies of all direct children for a given index
    def child_indecies(self, n1, n2):
        raise NotImplementedError
    
    # converts given circuit to image based representation
    def convert_to_image(self, *args):
        n1, n2 = args[0][0], args[0][1]
        raise NotImplementedError
    
# defines collections of all derivative circuits from seed circuit
class CircuitCollection(CircuitCollectionTemplate):
    
    def __init__(self, seed = None, num_qubits = 0, depth = None, reps = None):
        self.seed = seed # seed gate list
        self.num_qubits = num_qubits # number of qubits in circuit
        self.depth = depth if depth != None else len(seed)
        self.reps = reps if reps != None else list(np.ones(len(seed))) # number of times each gate is repeated

        # class internal globals
        self.circuits = [] # list of child circuits
        self.q_circuits = [] # list of qiskit circuits
        self.q_transpiled = [] # list of transpiled qiskit circuits
        self.images = [] # list of image based representations of circuits

        self.circuit_indexes = {} # dictionary of circuit indices (key: circuit, value: index tuple), O(1) lookup 

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
        # get index of circuit
        n1, n2 = self.circuit_indexes[tuple(gates)]

        return (int(n1), int(n2))
    
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
        # print(n1, n2)
        circuit = self.circuits[n1][n2]

        image = np.zeros((sum(self.reps), self.num_qubits)) # initialize image

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
        '''Generates all circuits of depth self.depth from seed circuit.
           Remove all duplicate circuits (keeps first occurance in order)'''

        # construct seed
        seed = []
        for i in range(len(self.seed)):
            for j in range(int(self.reps[i])):
                seed.append(self.seed[i])

        gen_start_time = time.time()
        # generate all circuit lists
        self.circuits = []
        for i in range(len(seed) - self.depth + 1, len(seed) + 1):
            it = unique_elem_it(itertools.permutations(seed, i))
            self.circuits.append(list(it))

            # indexes (maybe later hash the circuits)
            for j in range(len(self.circuits[-1])):
                self.circuit_indexes[self.circuits[-1][j]] = (i - (len(seed) - self.depth + 1), j)

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

        # FIXME: why did I think this was good ? lol
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


############ DATASET ############
# defines a dataset of circuits, with dynamic loading and saving for large datasets
# will randomize order of circuits, while ensuring that each circuit is in the same pickle file as its children
# inherits from CircuitCollectionTemplate to ensure that it is backwards compatible

# contains a section of the circuit dataset, used in conjunction with CircuitDataset
class CircuitSection(CircuitCollectionTemplate):

    def __init__(self, pickle_file):
        self.pickle_file = pickle_file

        self.circuits = [] # second list is bottom level
        self.q_circuits = []
        self.q_transpiled = []
        self.images = None # store as nested ragged tensor
        self.best_depths = []

        self.circuit_indexes = {}

        self.train_indecies = []
        self.val_indecies = []

        self.train_batches = []

class CircuitDataset(CircuitCollectionTemplate):

    def __init__(self, seed = None, num_qubits = 0, depth = None, reps = None, pickle_batch_size = None, pickle_root_folder = None):
        # assign all parameter to class
        self.pickle_batch_size = pickle_batch_size
        self.pickle_root_folder = pickle_root_folder

        # check if dir exists, if not create it
        if not os.path.exists(pickle_root_folder):
            os.mkdir(pickle_root_folder)

        self.seed = seed # seed gate list
        self.num_qubits = num_qubits # number of qubits in circuit
        self.depth = depth if depth != None else len(seed)
        self.reps = reps if reps != None else list(np.ones(len(seed))) # number of times each gate is repeated

        # class internal globals
        self.master_indexes = {} # dictionary of circuit indices (key: circuit, value: (pickle_index, index)), O(1) lookup (may not be necessary)
        self.generated_circuits = False
        self.built_circuits = False
        self.train_percent = 0.8 # percent of dataset to use for training
        self.train_set = True # if true, iterate through training set, else iterate through validation set
        self.batch_size = 1 # number of circuits to return per iteration

        self.pickle_list = [] # list of pickle files

        self.batch_number = 0 # total number of batches

    ##### GENERATION #####

    def generate_circuits(self):
        '''Generates all circuits of depth self.depth from seed circuit.
           Remove all duplicate circuits (keeps first occurance in order)'''

        print("Generating Circuits...")
        gen_start_time = time.time()

        # construct seed
        seed = []
        for i in range(len(self.seed)):
            for j in range(int(self.reps[i])):
                seed.append(self.seed[i])

        # generate top level
        circuits = list(unique_elem_it(itertools.permutations(seed, len(seed))))
        
        # shuffle list randomly
        np.random.shuffle(circuits)

        # split into batches of size pickle_batch_size (last batch may be smaller)
        circuits = [ [circuits[i:i + self.pickle_batch_size]] for i in range(0, len(circuits), self.pickle_batch_size)]

        indexes = [{} for i in range(len(circuits))]

        # create children for each batch
        # note this process may make duplicates, but this will be handled dynamically later
        for i in range(len(circuits)):
            indexes[i] = {circuit: (0, j) for j, circuit in enumerate(circuits[i][0])}

            for j in range(self.depth - 1): # continue until desired depth is reached
                circuits_to_compute = circuits[i][0] if j == 0 else copy.deepcopy(children)

                children = []
                for k in range(len(circuits_to_compute)): # generate children for each circuit
                    for l in range(len(circuits_to_compute[k])): # cut each circuit at each gate
                        child = copy.deepcopy(list(circuits_to_compute[k]))
                        child.pop(l)
                        children.append(tuple(child))

                # remove duplicates
                children = list(unique_elem_it(children))

                # add children to circuits
                circuits[i].append(children)

                # add indexes to dictionary
                for k in range(len(children)):
                    indexes[i][children[k]] = (j + 1, k)

        # for each batch, create a CircuitSection and pickle it
        for i in range(len(circuits)):
            # create section
            section = CircuitSection(self.pickle_root_folder + "/section_" + str(i) + ".p")

            # add pickle file to list
            self.pickle_list.append(section.pickle_file)

            # add circuits to section
            section.circuits = circuits[i]

            # add indexes to section
            section.circuit_indexes = indexes[i]

            # set master_indexes
            for circuit in section.circuit_indexes:
                # add circuit to master_indexes, if circuit already exists append to list
                if circuit in self.master_indexes:
                    self.master_indexes[circuit].append((i, section.circuit_indexes[circuit]))
                else:
                    self.master_indexes[circuit] = [(i, section.circuit_indexes[circuit])]

            # pickle section
            pickle.dump(section, open(section.pickle_file, "wb"))

        self.generated_circuits = True

        # print run time
        print("--- Generate: %s seconds ---" % (time.time() - gen_start_time))

    ##### BUILDING #####
    # build circuit with qiskit
    def build_circuit(self, gates = []):

        qc = QuantumCircuit(self.num_qubits, 0)

        for gate in gates:
            qc.cx(gate[0], gate[1])

        return qc
    
    def build_circuits(self):

        print("\nBuilding Circuits...")
        build_start_time = time.time()

        if not self.generated_circuits:
            raise Exception("build_circuits: No Circuits Generated!")
        
        # loop through sections
        for i in range(len(self.pickle_list)):

            # load section, build all circuits, and pickle it
            section = pickle.load(open(self.pickle_list[i], "rb"))

            for j in range(len(section.circuits)):
                section.q_circuits.append(list(map(self.build_circuit, section.circuits[j])))

            pickle.dump(section, open(section.pickle_file, "wb"))

            # print run time
            print("--- Build Section " + str(i) + ": %s seconds ---" % (time.time() - build_start_time))

        self.built_circuits = True

        # print run time
        print("--- Build Total: %s seconds ---" % (time.time() - build_start_time))

    ##### TRANSPILE #####

    # transpile circuits with n processes and transpile_args
    # trials specifies how many times to try transpile before choosing the best result (based on circuit depth)
    def transpile_circuits(self, n, trials, **transpile_args):

        print("\nTranspiling Circuits...")
        trans_start_time = time.time()

        if not self.built_circuits:
            raise Exception("transpile_circuits: No Circuits Built!")
        
        # loop through sections
        for l in range(len(self.pickle_list)):

            # load section
            section = pickle.load(open(self.pickle_list[l], "rb"))

            section.q_transpiled = [] # clear transpiled circuits

            transpiled_temp = []

            # compute first set of transpiled circuits
            for i in range(len(section.q_circuits)):
                section.q_transpiled.append(
                    parallel_map(transpile, section.q_circuits[i], num_processes = n, task_kwargs = transpile_args)
                )

            print("--- Transpile Section " + str(l) + " (Trial " + str(1) + "): %s seconds ---" % (time.time() - trans_start_time))

            # compute remaining trials and compare them to the first set, replacing them if they are better
            for i in range(trials - 1):
                for j in range(len(section.q_circuits)):

                    # Transpile circuits
                    transpiled_temp.append(
                        parallel_map(transpile, section.q_circuits[j], num_processes = n, task_kwargs = transpile_args)
                    )

                    # Compare circuits
                    for k in range(len(section.q_circuits[j])):
                        if (transpiled_temp[j][k].depth() < section.q_transpiled[j][k].depth()):
                            section.q_transpiled[j][k] = transpiled_temp[j][k]

                # print run time
                print("--- Transpile Section " + str(l) + " (Trial " + str(i + 2) + "): %s seconds ---" % (time.time() - trans_start_time))
                transpiled_temp = []

            # pickle section
            pickle.dump(section, open(section.pickle_file, "wb"))

            print()

        # print run time
        print("--- Transpile (" + str(trials) + "): %s seconds ---" % (time.time() - trans_start_time))

    ##### CONVERT TO IMAGES #####

    def convert_to_image(self, gate_list, fixed_size = True):

        if fixed_size:
            image = np.zeros((sum(self.reps), self.num_qubits)) # initialize image
        else:
            image = np.zeros((len(gate_list), self.num_qubits)) # initialize image

        # convert each gate to a column of the image
        for i, gate in enumerate(gate_list):
            # NOTE: set both to one (not preservering target/ctrl qubit information for now)
            image[i][gate[0]] = 1
            image[i][gate[1]] = 1

        # transpose image
        image = np.transpose(image)

        return tf.convert_to_tensor(image, dtype = tf.float32)

    # convert all circuits to image based representations
    def convert_to_images(self, fixed_size = True):

        print("\nConverting Circuits to Images...")
        convert_start_time = time.time()

        if not self.built_circuits:
            raise Exception("convert_to_images: No Circuits Built!")
        
        # loop through sections
        for l in range(len(self.pickle_list)):

            # load section
            section = pickle.load(open(self.pickle_list[l], "rb"))

            images = [] # temporary list of images

            # convert all circuits to images
            for i in range(len(section.circuits)):
                images.append(tf.ragged.stack(list(
                    map(self.convert_to_image, section.circuits[i], [fixed_size] * len(section.circuits[i]))
                    )))

            # convert to ragged tensor
            section.images = tf.ragged.stack(images)

            # pickle section
            pickle.dump(section, open(section.pickle_file, "wb"))

            # print run time
            print("--- Convert Section " + str(l) + ": %s seconds ---" % (time.time() - convert_start_time))

        # print run time
        print("--- Convert Total: %s seconds ---" % (time.time() - convert_start_time))

    ##### COMPUTE BEST CUTS #####

    # compute the optimal depth for each circuit in the dataset
    def compute_optimal_depths(self):
        
        print("\nComputing Optimal Depths...")
        convert_start_time = time.time()

        # loop through sections
        for l in range(len(self.pickle_list)):

            # load section
            self.current_section = pickle.load(open(self.pickle_list[l], "rb"))


            # loop through circuits
            for i in range(len(self.current_section.circuits) - 1): # exclude bottom level
                self.current_section.best_depths.append([])
                for j in range(len(self.current_section.circuits[i])):

                    # get circuit children
                    children = self.child_indecies(i, j)
                    # print(children)
                    # print(len(self.current_section.q_transpiled))

                    # loop through children and choose best depth
                    best_depth = 0
                    for child in children:
                        if self.current_section.q_transpiled[child[0]][child[1]].depth() > best_depth:
                            best_depth = self.current_section.q_transpiled[child[0]][child[1]].depth()
                    
                    self.current_section.best_depths[-1].append(best_depth)

            # print run time
            print("--- Compute Section " + str(l) + ": %s seconds ---" % (time.time() - convert_start_time))

            # pickle section
            pickle.dump(self.current_section, open(self.current_section.pickle_file, "wb"))

    ##### LOAD #####

    # load a section of the dataset
    def load_section(self, section_index):
        self.current_section = pickle.load(open(self.pickle_list[section_index], "rb"))

    ##### GETTERS #####
    
    # Takes a set of gates and returns the corresponding index in circuits/q_circuits/q_transpiled (provided that the circuit is in the current section)
    def gates_to_index(self, gates = []):
        # get index of circuit
        n1, n2 = self.current_section.circuit_indexes[tuple(gates)]

        return int(n1), int(n2)

    # returns the indecies of all direct children for a given index
    def child_indecies(self, n1, n2):

        children = itertools.combinations(self.current_section.circuits[n1][n2], len(self.current_section.circuits[n1][n2]) - 1)

        child_indecies = list(map(self.gates_to_index, children))

        return child_indecies
    
    #### SETTERS ####

    # set train percent and batch size
    def set_batches(self, train_percent, batch_size, internal_loops = 1):
        self.train_percent = train_percent
        self.batch_size = batch_size
        self.batch_number = 0

        # set train and validation indecies
        # loop through sections
        for l in range(len(self.pickle_list)):
            # load section
            section = pickle.load(open(self.pickle_list[l], "rb"))

            # get all possible index pairs
            indecies = []
            for i in range(len(section.circuits) - 1): # exclude bottom level
                for j in range(len(section.circuits[i])):
                    indecies.append((i, j))

                    # check if master_indexes has more than one entry for this circuit
                    if len(self.master_indexes[section.circuits[i][j]]) > 1 and self.master_indexes[section.circuits[i][j]].index((l, (i, j))) != 0: # by default keep the first occurence
                        indecies.pop(-1)

            # shuffle indecies
            np.random.shuffle(indecies)

            # split indecies into train and validation indecies
            section.train_indecies = indecies[:int(len(indecies) * self.train_percent)]
            section.val_indecies = indecies[int(len(indecies) * self.train_percent):]

            section.train_batches = []

            for i in range(internal_loops):
                # shuffle train indecies
                np.random.shuffle(section.train_indecies)

                # batch the train indecies
                section.train_batches += [section.train_indecies[i:i + self.batch_size] for i in range(0, len(section.train_indecies), self.batch_size)]

                # remove last batch if it is smaller than batch_size
                if len(section.train_batches[-1]) < self.batch_size:
                    section.train_batches.pop(-1)

            self.batch_number += len(section.train_batches)

            # pickle section
            pickle.dump(section, open(section.pickle_file, "wb"))

    # set if we are iterating through the training or validation set
    def set_train(self, train):
        self.train_set = train

    # # set the batch size
    # def set_batch_size(self, batch_size):
    #     self.batch_size = batch_size
    
    #### ITERATION STUFF ####

    def reset(self):
        self.current_section = pickle.load(open(self.pickle_list[0], "rb"))
        self.current_index = 0
        self.curent_section_index = 0

    def __iter__(self):
        self.current_section = pickle.load(open(self.pickle_list[0], "rb"))
        self.current_index = 0 # (index of the next batch to return) - 1
        self.current_section_index = 0 # index of the current section
        return self
    
    def __next__(self): # NOTE: batches smaller than batch_size are automatically thrown out during this process
        # self.current_index += self.batch_size # increment index
        self.current_index += 1 # increment index

        # # check if we need to load a new section
        # len_check = len(self.current_section.train_indecies) if self.train_set else len(self.current_section.val_indecies)

        if self.current_index >= len(self.current_section.train_batches): # if we need to load a new section
            self.current_section_index += 1

            if self.current_section_index >= len(self.pickle_list):
                raise StopIteration

            self.current_section = pickle.load(open(self.pickle_list[self.current_section_index], "rb"))
            self.current_index = 0

        # # create batch
        # if self.train_set:
        #     batch = copy.deepcopy(self.current_section.train_indecies[self.current_index - self.batch_size:self.current_index])
        # else:
        #     batch = copy.deepcopy(self.current_section.val_indecies[self.current_index - self.batch_size:self.current_index])

        return self.current_section.train_batches[self.current_index]