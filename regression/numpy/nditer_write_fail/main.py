import numpy as np

it = np.nditer(np.array([1, 2]), op_flags=["readwrite"])

assert it[0] == 1
