import numpy as np


def consume(x):
    return 1


a = np.array([[1, 2], [3, 4]])
row = a[0]
value = consume(row)

assert value == 1
