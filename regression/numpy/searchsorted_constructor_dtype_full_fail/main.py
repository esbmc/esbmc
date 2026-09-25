import numpy as np

i = np.searchsorted(np.full((3,), 5.7, dtype=int), 5.7)

assert i == 0
