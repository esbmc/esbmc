import numpy as np

x = np.random.beta(0.001, 0.001)

assert x >= 0.0
assert x <= 1.0
