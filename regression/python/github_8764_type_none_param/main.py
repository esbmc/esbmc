def m(hint) -> bool:
    return hint is type(None)


assert m(type(None))
assert not m(int)
