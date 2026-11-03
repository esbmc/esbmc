from functools import reduce


def sub(x: int, y: int) -> int:
    return x - y


assert reduce(sub, [10, 2, 1]) == 7
assert reduce(sub, [10, 2], 1) == 7
