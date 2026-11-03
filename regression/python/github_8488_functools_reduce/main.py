import functools
from functools import reduce


def mul(x, y):
    return x * y


def add(x: int, y: int) -> int:
    return x + y


assert reduce(mul, [2, 3, 4]) == 24
assert reduce(lambda a, b: a + b, [1, 2, 3]) == 6
assert functools.reduce(add, [5]) == 5
assert reduce(add, [], 7) == 7
assert reduce(mul, (2, 5), 3) == 30

try:
    reduce(add, [])
    assert False
except TypeError:
    pass
