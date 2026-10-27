import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
row = np.array([10, 20, 30])
it = np.nditer([row, a], op_flags=[["readwrite"], ["readonly"]])
