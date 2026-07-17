import numpy as np

# A numpy array is modeled as a fixed-size C array, not the heap PyListObject
# struct backing Python's `list`. Passing it into a user function used to
# either silently reinterpret the array's raw bytes as PyListObject fields
# (an unsound pointer-decay bug) or reject the call outright. The parameter
# now keeps its concrete array type and decays like a normal C array
# argument instead.
def first(a):
    return a[0]

a = np.array([1, 2, 3])
assert first(a) == 1
