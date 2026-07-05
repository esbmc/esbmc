import numpy as np

# Symbolic dividend: exercises the libm fmod call path (not constant folding).
x = nondet_float()
__ESBMC_assume(x >= 0.0 and x < 10.0)

r = np.fmod(x, 3.0)

# C/numpy fmod keeps the sign of the dividend, so for x in [0, 10) the
# remainder is in [0, 3).
assert r >= 0.0
assert r < 3.0
