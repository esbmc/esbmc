import numpy as np

floor_result = np.floor([2.9, -2.1, 0.0])
assert floor_result[0] == 2.0
assert floor_result[1] == -3.0
assert floor_result[2] == 0.0

fabs_result = np.fabs([-2.1, 2.1, -3.0])
assert fabs_result[0] == 2.1
assert fabs_result[1] == 2.1
assert fabs_result[2] == 3.0

trunc_result = np.trunc([1.8, -1.8, 10.123])
assert trunc_result[0] == 1.0
assert trunc_result[1] == -1.0
assert trunc_result[2] == 10.0
