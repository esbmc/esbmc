# Verification harness for numpy scalar math
# (src/python-frontend/models/numpy.py).
#
# sqrt/floor/ceil/fabs on concrete values chosen so the results are exact.
#
# ENSURES:
#   E1: sqrt of a perfect square is exact
#   E2: floor rounds toward negative infinity
#   E3: ceil rounds toward positive infinity
#   E4: fabs is the absolute value
import numpy as np

assert np.sqrt(4.0) == 2.0      # E1
assert np.floor(2.7) == 2       # E2
assert np.ceil(2.1) == 3        # E3
assert np.fabs(-3.0) == 3.0     # E4
