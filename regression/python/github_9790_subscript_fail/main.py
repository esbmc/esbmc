# isinstance on a dict subscript answers from the stored value's type (#9790)
d = {1: 2}
d[1] = "x"
assert isinstance(d[1], int)
