x: int = 5
s: str = "abc"
items: list = [1, 2, 3]
n: int
n = x + 1


def f(a: int, b: int) -> int:
    total: int = a + b
    return total


assert x == 5
assert s == "abc"
assert len(items) == 3
assert n == 6
assert f(2, 3) == 5
