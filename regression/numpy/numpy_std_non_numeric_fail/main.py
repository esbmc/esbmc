import numpy as np

a = np.array([[1, 2], ["x", "y"]])

assert np.std(a) == 0.0
