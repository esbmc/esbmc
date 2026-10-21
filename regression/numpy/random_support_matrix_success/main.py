import numpy as np

e = np.random.exponential(1.0, 3)
b = np.random.beta(2.0, 2.0, 3)
k = np.random.binomial(3, 0.5, 3)

assert np.min(e) >= 0.0
assert np.min(b) >= 0.0
assert np.max(b) <= 1.0
assert np.max(k) <= 3
