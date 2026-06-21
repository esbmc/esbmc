import numpy as np

lhs = np.array([[1.5], [2.5]])
rhs = np.array([3.0, 4.0])

sum_result = np.add(lhs, rhs)

assert sum_result[0][0] == 5.5
