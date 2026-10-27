from functools import reduce


def add(x: int, y: int) -> int:
    return x + y


d = {5: 1, 0: 2}
assert reduce(add, d) == 5
