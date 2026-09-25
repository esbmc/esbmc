import numpy as np

base = np.array([1.5, 2.5])
o = np.ones_like(base, dtype=bool)

assert o[0] == False
