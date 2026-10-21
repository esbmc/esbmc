import numpy as np

x = np.random.beta(2.0, 5.0)

assert x >= 0.0
assert x <= 1.0
