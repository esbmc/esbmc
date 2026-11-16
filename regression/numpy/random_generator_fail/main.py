import numpy as np

rng = np.random.default_rng(0)
value = rng.integers(0, 4)
assert value >= 0
assert value < 4
