import numpy as np

caught = 0
try:
    a = np.linalg.inv(np.array(5.0))
except np.linalg.LinAlgError:
    caught += 1
try:
    w = np.linalg.eigvals(np.array([1.0, 2.0]))
except np.linalg.LinAlgError:
    caught += 1
try:
    s = np.linalg.svd(np.array([]), compute_uv=False)
except np.linalg.LinAlgError:
    caught += 1
try:
    x = np.linalg.solve(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), np.array([1.0, 2.0]))
except np.linalg.LinAlgError:
    caught += 1
try:
    d = np.linalg.det([[1.0, 2.0], [3.0]])
except ValueError:
    caught += 1
try:
    y = np.linalg.solve(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([1.0, 2.0, 3.0]))
except np.linalg.LinAlgError:
    caught += 100
except ValueError:
    caught += 1
assert caught == 6
