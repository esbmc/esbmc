def m(hint) -> bool:
    return hint is type(None)


t = type(None)
assert t is type(None)
assert t is not int
assert m(type(None))
assert not m(int)
x = None
assert type(x) is type(None)
