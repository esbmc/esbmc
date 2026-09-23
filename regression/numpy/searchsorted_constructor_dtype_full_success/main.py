import numpy as np

i = np.searchsorted(np.full((3,), 5, dtype=int), 5)

assert i == 0
