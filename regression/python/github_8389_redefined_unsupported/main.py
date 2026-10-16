def is_a(v, h) -> bool:
    return isinstance(v, h)


assert is_a(1, int)


def is_a(v, h) -> bool:
    return not isinstance(v, h)


assert is_a("a", int)
