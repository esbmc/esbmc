import numpy as np

x = np.random.choice([1, 2, 3], replace=False)

assert x >= 1
assert x <= 3
