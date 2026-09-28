import numpy as np

result = np.linalg.norm(np.array([1.0, -2.0, 3.0]), 1)

assert result == 6.0
