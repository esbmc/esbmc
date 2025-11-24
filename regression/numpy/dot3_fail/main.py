import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.dot(a, b)
assert result == 31  # should fail since 1*4 + 2*5 + 3*6 = 32
