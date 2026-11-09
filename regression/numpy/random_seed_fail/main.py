import numpy as np

np.random.seed(3)
np.random.seed(None)
value = np.random.random()
assert value >= 0.0
assert value < 1.0
