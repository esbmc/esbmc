import numpy as np

k = np.random.binomial(10, 0.3)

assert k >= 0
assert k <= 10
