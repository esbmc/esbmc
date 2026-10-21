def is_a(v, h) -> bool:
    return isinstance(v, h)


assert is_a(1, int)
assert is_a(1, (int, str))
