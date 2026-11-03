from functools import reduce


def add(x: int, y: int) -> int:
    return x + y


xs: list[int] = [1]
xs.pop()
total = reduce(add, xs)
assert total == 0
