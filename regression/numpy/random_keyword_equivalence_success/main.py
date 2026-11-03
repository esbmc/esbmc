import numpy as np

x = np.random.normal(scale=0.0, loc=1.0)
k = np.random.binomial(p=1.0, n=3)

assert x == 1.0
assert k == 3
