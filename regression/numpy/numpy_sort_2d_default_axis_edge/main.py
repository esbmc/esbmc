import numpy as np

a = np.array([[3, 1], [4, 2]])
result = np.sort(a)  # Default axis=-1 (sorts each row)

assert result[0, 0] == 1
assert result[1, 0] == 2
