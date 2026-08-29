import numpy as np

# np.greater(...) used directly (nested) as np.where()'s condition must be
# evaluated, not have its own first argument silently substituted for its
# unevaluated result.
a = [1, 2, 3]
b = [1, 5, 3]
r = np.where(np.greater(b, a), [10, 20, 30], [1, 2, 3])
assert r[0] == 1
assert r[1] == 20
assert r[2] == 3
