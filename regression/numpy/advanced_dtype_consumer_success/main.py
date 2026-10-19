import numpy as np

a = np.linspace(0, 3, 4, dtype=int)
assert np.sum(a) == 6
assert max(a) == 3
assert a.mean() == 1.5
