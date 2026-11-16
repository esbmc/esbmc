import math
import numpy as np

x = nondet_float()
__ESBMC_assume(x == 0.0 and math.copysign(1.0, x) == -1.0)
assert math.copysign(1.0, np.log1p(x)) == 1.0
