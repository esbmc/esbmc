import numpy as np

result = np.linalg.norm(np.array([[1.0, 2.0], [3.0, 4.0]]), axis=0)

assert result[0] == 1.0
