import numpy as np

arr = np.array([1,2,3,-4])

assert arr[0] == 1
assert arr[1] == 2
assert arr[2] == 3
assert arr[3] == -4


matrix = np.array([[1,2], [3,4]])
assert matrix[0][0] == 1
assert matrix[0][1] == 2
assert matrix[1][0] == 3
assert matrix[1][1] == 4
