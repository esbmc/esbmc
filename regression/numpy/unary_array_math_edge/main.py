import numpy as np

floor_result = np.floor([0.0, -0.0, 1.999999, -1.999999])
assert floor_result[0] == 0.0
assert floor_result[1] == 0.0
assert floor_result[2] == 1.0
assert floor_result[3] == -2.0

fabs_result = np.fabs([0.0, -0.0, -1e-100, 1e-100])
assert fabs_result[0] == 0.0
assert fabs_result[1] == 0.0
assert fabs_result[2] == 1e-100
assert fabs_result[3] == 1e-100

trunc_result = np.trunc([0.999999, -0.999999, 1234567.89, -1234567.89])
assert trunc_result[0] == 0.0
assert trunc_result[1] == 0.0
assert trunc_result[2] == 1234567.0
assert trunc_result[3] == -1234567.0
