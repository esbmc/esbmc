def default_int(x: int, y: int = 2) -> int:
    return x + y


x: int = default_int(1)
assert x == 3
