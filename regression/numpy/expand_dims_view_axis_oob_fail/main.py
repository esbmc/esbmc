import numpy as np

a = np.array([1, 2, 3])
v = np.expand_dims(a, 3)

assert v[0] == 1
