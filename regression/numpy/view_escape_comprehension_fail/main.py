import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
rows = [a[i] for i in range(2)]
rows[0][1] = 9
assert a[0][1] == 9
