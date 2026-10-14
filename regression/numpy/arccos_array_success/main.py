import numpy as np

values = [0.0, 1.0]
result = np.arccos(values)

assert result[0] >= 0.0
assert result[0] <= 3.141593
assert result[1] < 1e-6
