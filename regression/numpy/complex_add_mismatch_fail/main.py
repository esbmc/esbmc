import numpy as np

a = np.array([1 + 2j, 3j])
b = np.array([1j, 2])
c = a * b
assert c[0] == 1 + 2j
