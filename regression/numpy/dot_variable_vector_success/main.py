import numpy as np

# np.dot(a, b) used as a bare expression statement (its result is never
# assigned) with both operands read from a variable, not a literal. This is
# the exact process-crash repro from the roadmap: it must not abort the
# GOTO conversion regardless of whether the caller keeps the result.
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.dot(a, b)
