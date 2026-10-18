class Number:
    pass


def is_a(v, h) -> bool:
    return isinstance(v, h)


assert is_a(1, int)
int = Number
assert is_a(Number(), int)
