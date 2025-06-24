import numpy as np
b = np.transpose([[-1, 2], [3, -4]])
<<<<<<< HEAD
<<<<<<< HEAD
assert b[0][0] == -1
assert b[1][1] == -4
=======
assert b[0][0] == 1   # Wrong! Should be -1
assert b[1][1] == 4   # Wrong! Should be -4
>>>>>>> 96cf203b1 ([python] added test cases for numpy transpose)
=======
assert b[0][0] == -1
assert b[1][1] == -4
>>>>>>> 468bb9ad1 ([numpy] disallow 3D or higher-dimensional arrays (#2492))
