import numpy as np
import random

r = np.arange(5, dtype=int)
w = r[1:]
assert min(w) == 2
