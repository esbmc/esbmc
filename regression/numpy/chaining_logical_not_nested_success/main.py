import numpy as np

# np.equal(...) used directly (nested) as np.logical_not()'s argument must be
# evaluated, not have its own first argument silently substituted for its
# unevaluated result.
a = [1, 2, 3]
b = [1, 5, 3]
r = np.logical_not(np.equal(a, b))
assert r[0] == False
assert r[1] == True
assert r[2] == False
