import numpy as np

a = np.array([42])
b = [x for x in a]

assert b[0] == 42
