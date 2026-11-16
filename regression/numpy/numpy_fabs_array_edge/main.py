import math
import numpy as np

r = np.fabs(np.array([-0.0, math.nan, -math.inf, -5e-324, -1.7976931348623157e308]))
assert r[0] == 0.0 and math.copysign(1.0, r[0]) == 1.0
assert r[1] != r[1] and r[2] == math.inf
assert r[3] == 5e-324 and r[4] == 1.7976931348623157e308
