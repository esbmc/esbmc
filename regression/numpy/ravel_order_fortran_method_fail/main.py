import numpy as np

a = np.array([[1, 2], [3, 4]])
a.ravel('F')[0] = 99
