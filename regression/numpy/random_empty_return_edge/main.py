import numpy as np

a = np.random.exponential(size=0)

assert np.sum(a) == 0.0
assert len(a.tolist()) == 0
