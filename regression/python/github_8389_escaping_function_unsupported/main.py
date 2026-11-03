def is_a(v, h) -> bool:
    return isinstance(v, h)


check = is_a
assert is_a(1, int)
assert not check(1, str)
