import numpy as np

rng = np.random.default_rng(3)
x = rng.choice([1, 10], p=[0.25, 0.7500000001])
y = np.random.choice([1, 10], p=[0.5, 0.5])

assert x == 1 or x == 10
assert y == 1 or y == 10
