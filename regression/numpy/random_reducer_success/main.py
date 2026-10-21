import numpy as np

a = np.random.uniform(1.0, 2.0, size=3)
doubled = a * 2.0

assert np.sum(a) >= 3.0
assert np.mean(a) <= 2.0
assert np.max(a) <= 2.0
assert np.min(a) >= 1.0
assert doubled[2] <= 4.0
