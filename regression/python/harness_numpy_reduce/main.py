# Verification harness for numpy reductions
# (src/python-frontend/models/numpy.py).
#
# sum/prod/max/min reduce a numeric list to a scalar. The numpy handler
# supports constant inputs, so this harness is concrete.
#
# ENSURES:
#   E1: sum adds every element
#   E2: prod multiplies every element
#   E3: max returns the largest element
#   E4: min returns the smallest element
import numpy as np

assert np.sum([1, 2, 3]) == 6       # E1
assert np.prod([2, 3, 4]) == 24     # E2
assert np.max([1, 5, 2]) == 5       # E3
assert np.min([4, 1, 3]) == 1       # E4
