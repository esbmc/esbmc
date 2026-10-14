import numpy as np

# np.greater(...) and np.equal(...) used directly (nested) as
# np.logical_and()'s arguments must both be evaluated, not have their raw
# arguments silently substituted for their unevaluated results. Neither
# logical_and nor logical_or had chaining coverage before this test.
a = [1, 2, 3]
b = [1, 5, 3]
r = np.logical_and(np.greater(b, a), np.equal(a, a))
assert r[0] == False
assert r[1] == True
assert r[2] == False
