import numpy as np


a = np.array([[1, 2], [3, 4]])
t = np.transpose(a)

assert consume_unknown(t) == 1
