import numpy as np

o = np.zeros(1, dtype=int)
np.fmod(np.array([7.5]), 2.0, out=o)
