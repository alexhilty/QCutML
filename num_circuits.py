import numpy as np

def num_circuits(n, depth = None):
    # compute number of gates (just going one direction, so cancellations are not possible)
    g = int(n*(n-1)/2)

    depth = depth if depth != None else g

    # compute number of possible circuits
    sum = 0
    for i in range(g - depth + 1,g+1):
        sum += np.math.factorial(g) / np.math.factorial(g-i)

    return int(sum)

print(num_circuits(4, 2))