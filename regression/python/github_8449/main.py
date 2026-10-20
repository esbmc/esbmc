def f() -> int:
    return 1


x = f()
try:
    y = h()
except NameError:
    y = 2


def h() -> int:
    return 5


assert x == 1
assert y == 2
assert h() == 5
