import numpy as np

k = np.random.poisson(3.5)

assert k >= 0
assert k % 1 == 0
