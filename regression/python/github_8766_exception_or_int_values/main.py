def m(v):
    return v


e = m(ValueError("x"))
assert isinstance(e, ValueError)
assert m(1) == 1
