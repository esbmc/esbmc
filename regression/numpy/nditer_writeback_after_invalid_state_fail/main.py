import numpy as np

a = np.array([1, 2])
it = np.nditer(a, op_flags=[["readwrite"]])
while not it.finished:
    it.iternext()
it[0][...] = 5
