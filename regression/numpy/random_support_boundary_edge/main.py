import numpy as np

assert np.random.geometric(1.0) == 1
assert np.random.binomial(5, 1.0) == 5
assert np.random.weibull(0.0) == 0.0
