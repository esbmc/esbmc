import numpy as np

nan = float("nan")
raised = 0
try:
    np.random.poisson(nan)
except ValueError:
    raised += 1
try:
    np.random.binomial(1, nan)
except ValueError:
    raised += 1
try:
    np.random.geometric(-0.0)
except ValueError:
    raised += 1
assert raised == 3
