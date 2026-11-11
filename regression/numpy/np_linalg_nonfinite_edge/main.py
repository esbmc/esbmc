import numpy as np

caught = 0
try:
    w = np.linalg.eigvals(np.array([[float("nan"), 1.0], [1.0, 2.0]]))
except np.linalg.LinAlgError:
    caught += 1
try:
    s = np.linalg.svd(np.array([[float("inf"), 1.0], [1.0, 2.0]]), compute_uv=False)
except np.linalg.LinAlgError:
    caught += 1
x = nondet_float()
try:
    v = np.linalg.eigvals(np.array([[x, 1.0], [1.0, 2.0]]))
    assert x == x
except np.linalg.LinAlgError:
    assert x != x or x == float("inf") or x == float("-inf")
assert caught == 2
