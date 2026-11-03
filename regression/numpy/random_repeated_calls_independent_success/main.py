import numpy as np

draws = 0
for i in range(3):
    v = np.random.poisson(1.0)
    assert v >= 0
    draws = draws + 1

assert draws == 3
