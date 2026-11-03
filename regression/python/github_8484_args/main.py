def inc(x: int) -> int:
    return x + 1


def dbl(x: int) -> int:
    return x * 2


def apply(k: int, v: int) -> int:
    return ops[k](v)


ops = [inc, dbl]
assert apply(0, 5) == 6
assert apply(1, 5) == 10
