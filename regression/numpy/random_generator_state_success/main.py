import numpy as np

rng = np.random.default_rng(None)
alias = rng
matrix = rng.integers(0, 4, size=(2, 2), dtype=np.int8)
scalar = alias.random()
unit = alias.random(size=1)
fixed = alias.integers(1, 1, size=2, endpoint=True)

assert matrix.shape == (2, 2)
assert matrix[0][0] >= 0
assert matrix[0][0] < 4
assert scalar >= 0.0
assert scalar < 1.0
assert unit.shape == (1,)
assert fixed[0] == 1
assert fixed[1] == 1
