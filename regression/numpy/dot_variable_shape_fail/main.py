import numpy as np

# Both 1-D operands are variables (not literals) with incompatible lengths.
# Must produce the existing explicit broadcast diagnostic, not a crash and
# not a silently wrong dot product.
a = np.array([1, 2, 3])
b = np.array([4, 5])

result = np.dot(a, b)

assert result == 0
