import numpy as np

a = np.array([])
b = a.reshape(0, 3)
np.argmin(b, axis=0)
