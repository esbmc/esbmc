import numpy as np

# A small, constant-argument np.arange(...) must be cheap to verify: this
# used to route through the operational model's while-loop implementation
# (models/numpy.py's arange()), which is disproportionately expensive to
# symbolically execute even for three elements.
a = np.arange(3)
assert len(a) == 3
assert a[0] == 0
assert a[1] == 1
assert a[2] == 2
