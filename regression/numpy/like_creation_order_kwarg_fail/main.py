import numpy as np

base = np.array([1, 2], dtype=int)
np.zeros_like(base, order="F")
