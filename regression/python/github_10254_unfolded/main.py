# Issue #10254: calls the int fold gives up on still get CPython's values.
def p(a: int, b: int) -> float:
    return a ** b


def band(a: int, b: int) -> int:
    return a & b


def fd(a: int, b: int) -> int:
    return a // b


def aug(a: int, b: int) -> int:
    a //= b
    return a


def ab(a: int) -> int:
    return abs(a)


def ng(a: int) -> int:
    return -a


def cat(s: str) -> str:
    s += "b"
    return s


assert p(2, -1) == 0.5
assert p(1, 65) == 1.0
assert band(6, 3) == 2
assert ab(-5) == 5
assert ng(5) == -5
assert cat("a") == "ab"

caught = 0
try:
    fd(1, 0)
except ZeroDivisionError:
    caught += 1
try:
    aug(1, 0)
except ZeroDivisionError:
    caught += 1
assert caught == 2
