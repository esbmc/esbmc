import numpy as np

#array A with (2,) shape
A = np.array([1,2])

#array B with (3,) shape
B = np.array([10,20,30])

# This sum generates error, as the shape (2,) and (3,) are not compatible
C = np.add(A,B)

#assert C[0] == 11
#assert C[1] == 22