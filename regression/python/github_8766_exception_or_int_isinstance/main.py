def m(v, hint) -> bool:
    return isinstance(v, hint)


assert m(ValueError("x"), Exception)
assert not m(1, Exception)
