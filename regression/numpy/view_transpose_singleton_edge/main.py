import numpy as np

a = np.array([[5]])
t = a.T

assert t[0][0] == 5
