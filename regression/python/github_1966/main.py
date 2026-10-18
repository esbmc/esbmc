def half(a: int) -> int:
    return a / 2


def scaled(a: int) -> int:
    return half(a) * 1.5


assert half(4) == 2
assert scaled(4) == 3
