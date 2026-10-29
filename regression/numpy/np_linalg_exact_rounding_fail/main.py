import numpy as np

n = np.linalg.norm(np.array([0.1, 0.2, 0.3]))
assert n == 0.37416573867739417, "rounding"
