import numpy as np

# np.equal() called with the wrong number of arguments, nested as another
# numpy call's argument, must raise its own explicit diagnostic once
# evaluated -- not be misreported as exceeding the chaining depth (this is
# only one level deep), and not be silently treated as a false operand.
a = [1, 2, 3]
r = np.logical_not(np.equal(a))
assert len(r) == 3
