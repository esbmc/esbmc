import numpy as np

idx = np.argsort(np.linspace(2.9, 2.0, 3, dtype=int))

assert idx[0] == 2
