import numpy as np

rng = np.random.default_rng(11)
positional = rng.integers(0, 3, 2)
keyword = rng.integers(low=0, high=3, size=2)

assert positional.shape == (2,)
assert keyword.shape == (2,)
assert positional[0] >= 0
assert positional[0] < 3
assert keyword[0] >= 0
assert keyword[0] < 3
