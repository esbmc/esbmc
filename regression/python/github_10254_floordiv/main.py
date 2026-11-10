# Issue #10254: a folded call floors // and % like CPython.
def q(a: int, b: int) -> int:
    return a // b


def m(a: int, b: int) -> int:
    return a % b


assert q(-7, 2) == -4
assert q(7, -2) == -4
assert q(7, 2) == 3
assert m(-7, 2) == 1
assert m(7, -2) == -1
