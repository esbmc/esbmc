def m(v):
    return v


def is_a(v, hint) -> bool:
    return isinstance(v, hint)


m(ValueError("x"))
m(1)
assert is_a(ValueError("x"), Exception)
assert not is_a(1, Exception)
