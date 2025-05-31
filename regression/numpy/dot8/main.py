import numpy as np

# Create two 3D arrays of shape (2, 3, 3)
a = np.array([
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]],

    [[10, 11, 12],
     [13, 14, 15],
     [16, 17, 18]]
])

b = np.array([
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]],

    [[1, 2, 3],
     [0, 1, 0],
     [3, 2, 1]]
])

result = np.dot(a, b)
