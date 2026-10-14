import numpy as np

counter = [0]


def get_array():
    counter[0] = counter[0] + 1
    return np.array([1, 2, 3])


y = get_array()

assert y[0] == 1
assert counter[0] == 1
