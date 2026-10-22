g = 1


def bump() -> int:
    global g
    g = 10
    return 5


def twice(n: int) -> int:
    return n * 2


x = g * 100 + twice(bump())
assert x == 110
g = 1
assert g < bump()
