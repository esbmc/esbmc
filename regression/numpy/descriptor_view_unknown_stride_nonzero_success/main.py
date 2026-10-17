import numpy as np

step = nondet_int()
__ESBMC_assume(step != 0)
a = np.array([1, 2, 3])
part = a[::step]
assert len(part) >= 1
assert len(part) <= 3
