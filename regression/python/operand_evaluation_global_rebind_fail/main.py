g = 1


def bump() -> int:
    global g
    g = 10
    return 5


x = g + bump()
assert x == 15
