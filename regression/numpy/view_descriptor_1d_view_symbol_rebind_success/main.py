import numpy as np

# v was retyped to a pointer when it was first created as a 1-D view; that
# retype is never undone (its DECL already committed pointer_typet), but
# rebinding v to a fresh array still works soundly: the array-to-pointer
# decay already used elsewhere for pointer-typed targets applies here too,
# and clear_numpy_view_copy drops the stale view-length tracking from the
# old `a[1:3]` binding so len(v) reports the new array's real size, not the
# old view's.
a = np.array([1, 2, 3, 4])
v = a[1:3]
v = np.array([5, 6, 7])

assert len(v) == 3
assert v[0] == 5
assert v[1] == 6
assert v[2] == 7
