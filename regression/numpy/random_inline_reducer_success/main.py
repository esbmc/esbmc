import numpy as np

assert np.sum(np.random.random(3)) < 3.0
assert np.max(np.random.randint(0, 4, size=(2, 2))) < 4
assert np.mean(np.random.uniform(1.0, 2.0, 2)) >= 1.0
assert np.any(np.random.randint(1, 3, size=2))
