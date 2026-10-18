from functools import reduce


def mul(x, y):
    return x * y


xs = [2] + [3]
p = reduce(mul, xs)
assert isinstance(p, int)
assert p == 6
