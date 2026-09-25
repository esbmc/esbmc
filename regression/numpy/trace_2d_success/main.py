import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
t = np.trace(a)

assert t == 15
