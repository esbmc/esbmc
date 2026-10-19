import numpy as np

a = np.array([1 + 2j, 3j])
b = a * 2
assert b[0] == 2 + 4j
