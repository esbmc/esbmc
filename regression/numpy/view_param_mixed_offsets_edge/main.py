import numpy as np

def sum_pair(r1, r2):
    result = r1[0] + r2[0]
    return result

a = np.array([10, 20, 30])
v1 = a[0:]
v2 = a[1:]
total = sum_pair(v1, v2)
assert total == 30
