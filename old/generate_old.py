from Circuit_old import Circuit as c

# Initialize a 3 qubit circuit
gates = [(0, 1), (1, 2), (0, 2)]
circuit = c(None, gates, 3)

# generate all children
# circuit.generate_children_depth(-1)
circuit.generate_all_children()
circuit.build_all_circuits()
circuit.transpile_all(initial_layout = [0, 1, 2], coupling_map = [[0, 1], [1, 2]], optimization_level=2)

def func(node):
    print(node.qc.draw())
    # print(node.qc_t.draw())

circuit.traverse(func)

# transpile all
# circuit.transpile(-1, coupling_map = [[0, 1], [1, 2], [2, 3]], optimization_level=2) # linear nearest neighbor coupling map

# print(circuit.qc_t.draw())
# print(circuit.qc_t.depth())