import math
import numpy as np

x = nondet_float()
if x >= 1.0 and x <= 2.0:
    assert np.hypot(x, 1.0) >= 1.0 and np.hypot(x, 1.0) <= 3.0
    assert math.hypot(x, 1.0) >= 1.0 and math.hypot(x, 1.0) <= 3.0
    assert np.arctan2(x, 1.0) > 0.0 and np.arctan2(x, 1.0) < 1.571
    assert math.atan2(x, 1.0) > 0.0 and math.atan2(x, 1.0) < 1.571
    assert np.arctan2(-x, -1.0) < -1.57 and np.arctan2(-x, -1.0) > -3.142
