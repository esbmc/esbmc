import numpy as np

a = np.array([])
b = np.array([])
c = np.concatenate([a, b])

assert len(c) == 0
