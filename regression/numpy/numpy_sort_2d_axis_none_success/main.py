import numpy as np

a = np.array([[3, 1], [4, 2]])
result = np.sort(a, axis=None)  # Flattens and sorts: [1, 2, 3, 4]

assert len(result) == 4
assert result[0] == 1
assert result[3] == 4
