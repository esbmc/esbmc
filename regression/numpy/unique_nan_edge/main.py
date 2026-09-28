import numpy as np

u = np.unique(np.array([float("nan")]))

assert len(u) == 1
