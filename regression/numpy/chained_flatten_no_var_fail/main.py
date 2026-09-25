import numpy as np

a = np.full((2, 2), 5).flatten()

assert a[2] == 99
