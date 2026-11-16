# isinstance on a dict subscript answers from the stored value's type (#9790)
c = {1: 2, 2: "x"}
assert isinstance(c[1], int)
assert not isinstance(c[2], int)
d = {1: 2}
d[1] = "x"
assert isinstance(d[1], str)
assert not isinstance(d[1], int)
