# The calls pass a list and an int in different positions, so even a constant
# index names no single element type: refused rather than misread (#8472).
def first_is_list(a):
    return isinstance(a[0], list)


assert first_is_list([[1], 2])
assert not first_is_list([3, [4]])
