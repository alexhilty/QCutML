from qiskit import *
from qiskit.tools.jupyter import *
from qiskit.visualization import *

from qiskit_aer import AerSimulator
from qiskit_ibm_provider import IBMProvider


# Class to define a circuit
# ASSUMPTIONS: no gates are repeated, all CNOT gates, all target qubits are greater than control qubits
class Circuit:
    
    def __init__(self, parent = None, gates = [], num_qubits = 0):
        self.parent = parent # parent circuit
        self.gates = gates # gate list in circuit order
        self.num_qubits = num_qubits # number of qubits in circuit
        
        # class internal variables
        self.children = [] # list of child circuits
        self.qc = None # qiskit circuit
        self.qc_t = None # transpiled qiskit circuit

        self.generated_children = False
        self.built_circuit = False
        self.transpiled = False

    # return the number of gates in the circuit
    def num_gates(self):
        return len(self.gates)

    # based on the current configuration, generate all possible children
    def generate_children(self):
        # clear children
        self.children = []
        self.generated_children = True

        # if there is only one gate, then there are no children
        if len(self.gates) == 1:
            return

        # otherwise, generate all possible children
        for i in range(len(self.gates)):
            child_gates = self.gates.copy()
            child_gates.pop(i)
            self.children.append(Circuit(self, child_gates, self.num_qubits))

    ####### RECURSIVE FUNCTIONS #######

    # generate children up to depth d (d = -1 generates all, d = 1 generates self and first children, etc.)
    def generate_children_rec(self, d = 0):
        if d == 0:
            self.generate_children()
            return
        else:
            self.generate_children()
            for child in self.children:
                child.generate_children_depth(d-1)

    # transpile the circuit cascading to number of depths -> d = 0 transpiles self only, d = 1 transpiles self and children, etc., d = -1 transpiles all
    def transpile_rec(self, d = 0, **transpile_args):
        if not self.built_circuit:
            print("transpile: No Circuit Built!")

        if d == 0:
            self.qc_t = transpile(self.qc, **transpile_args)
        else:
            self.qc_t = transpile(self.qc, **transpile_args)
            for child in self.children:
                child.transpile(d-1, **transpile_args)

        self.transpiled = True

    # build circuit with qiskit to depth
    def build_circuit_rec(self, d = 0):

        if d == 0:
            self.qc = QuantumCircuit(self.num_qubits, 0)

            for gate in self.gates:
                self.qc.cx(gate[0], gate[1])
        
        else:
            self.qc = QuantumCircuit(self.num_qubits, 0)

            for gate in self.gates:
                self.qc.cx(gate[0], gate[1])

            for child in self.children:
                child.build_circuit(d-1)

        self.built_circuit = True

    ####### BFS FUNCTIONS #######
    
    def generate_all_children(self):
        self.traverse(lambda node: node.generate_children_rec(d = 0))

    def build_all_circuits(self):
        self.traverse(lambda node: node.build_circuit_rec(d = 0))

    def transpile_all(self, **transpile_args):
        self.traverse(lambda node: node.transpile_rec(d = 0,**transpile_args))

    # breadth first traversal of the circuit tree
    def traverse(self, func):
        queue = [self]

        while len(queue) > 0:
            cur_node = queue.pop(0)

            func(cur_node)

            for child in cur_node.children:
                queue.append(child)
    