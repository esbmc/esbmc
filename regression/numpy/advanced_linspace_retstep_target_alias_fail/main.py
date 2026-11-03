import numpy as np

start = 0
start, step = np.linspace(start, 1, 5, retstep=True)
assert step == 0.25 and start[4] == 1.0
