import numpy as np

# NumPy's det is exp(log|u11| + log|u22|) of the LU factors: -2.0000000000000004.
x = np.linalg.det(np.array([[1, 2], [3, 4]]))
assert x == -2
