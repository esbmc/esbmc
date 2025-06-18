import numpy as np
b = np.transpose([[0.000001, 0.000002], [0.000003, 0.000004]])
assert abs(b[0][0] - 0.000001) < 1e-10
assert abs(b[0][1] - 0.000003) < 1e-10
assert abs(b[1][0] - 0.000002) < 1e-10
assert abs(b[1][1] - 0.000004) < 1e-10
