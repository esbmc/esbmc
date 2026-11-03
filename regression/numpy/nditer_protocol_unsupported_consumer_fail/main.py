import numpy as np

a = np.array([5, 6, 7])
it = np.nditer(a)
vals = list(it)
assert vals[2] == 7
