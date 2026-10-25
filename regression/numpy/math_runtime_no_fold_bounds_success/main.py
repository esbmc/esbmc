import numpy as np

a = np.array([0.5, 1.0])
b = np.sin(a)
assert b[0] > 0.479 and b[1] < 0.842
