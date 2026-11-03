import numpy as np

a = np.random.binomial(3, 0.5, size=4)

assert a[-1] <= 3
assert a[-4] >= 0
