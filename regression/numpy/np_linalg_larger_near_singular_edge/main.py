import numpy as np

# Nearly singular but not singular: NumPy inverts it, without LinAlgError.
a = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0000000001],
              [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
b = np.linalg.inv(a)
assert len(b) == 4 and len(b[3]) == 4
