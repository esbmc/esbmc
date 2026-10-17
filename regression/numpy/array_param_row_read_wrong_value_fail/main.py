import numpy as np

# A row returned through a local alias is a view of the caller's array: the
# call is folded into the caller, so `row` here is `a[0]`.
def get_row(a):
    row = a[0]
    return row

a = np.array([[1, 2], [3, 4]])
row = get_row(a)
assert row[0] == 1
assert row[1] == 4
