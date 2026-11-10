import numpy as np

x = nondet_float()
caught = False
try:
    w, v = np.linalg.eig(x)
except np.linalg.LinAlgError:
    caught = True
assert caught
