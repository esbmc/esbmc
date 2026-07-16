import numpy as np

a = np.array([1, 2, 3])
b = a[10::2]

assert len(b) == 0
