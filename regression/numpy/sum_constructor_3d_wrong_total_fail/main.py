import numpy as np

a = np.ones((2, 2, 2))
b = np.sum(a)

assert b == 4
