def is_a(v, h) -> bool:
    if v == 0:
        h = str
    return isinstance(v, h)


assert is_a(1, int)
assert not is_a(0, int)
