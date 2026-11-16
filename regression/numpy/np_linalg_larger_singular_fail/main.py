import numpy as np

a = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 0.0, 1.0], [1.0, 2.0, 3.0, 4.0],
              [2.0, 0.0, 1.0, 0.0]])
b = np.linalg.inv(a)
