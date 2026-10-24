g = 1
box = [1]


def bump() -> int:
    global g
    g = 10
    return 5


def poke() -> int:
    box[0] = 10
    return 5


def diff() -> int:
    return g - bump()


x = g + bump()
assert x == 6
assert g == 10

y = box[0] + poke()
assert y == 6

g = 1
assert diff() == -4
