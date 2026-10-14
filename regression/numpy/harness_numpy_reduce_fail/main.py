# Falsification harness for numpy reductions
# (src/python-frontend/models/numpy.py).
#
# np.sum([1, 2, 3]) is 6, not 7, so the wrong total must be falsifiable.
#
# WRONG PROPERTY (expected to be falsified):
#   F1: np.sum([1, 2, 3]) == 7.  The sum is 6.
import numpy as np

assert np.sum([1, 2, 3]) == 7  # F1 — falsifiable (sum is 6)
