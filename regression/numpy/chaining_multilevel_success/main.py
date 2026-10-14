import numpy as np

# Chaining more than one level (np.logical_not(np.logical_not(...))) is not
# rejected: the chain-resolution recursion handles depths well beyond what
# realistic code needs.
a = [1, 2, 3]
b = [1, 5, 3]
r = np.logical_not(np.logical_not(np.equal(a, b)))
assert r[0] == True
assert r[1] == False
assert r[2] == True
