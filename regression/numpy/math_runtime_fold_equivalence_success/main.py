import numpy as np

a = np.array([2.0])
assert np.exp(a)[0] > 7.389 and np.exp(a)[0] < 7.3891
assert np.exp([2.0])[0] > 7.389 and np.exp(2.0) > 7.389
