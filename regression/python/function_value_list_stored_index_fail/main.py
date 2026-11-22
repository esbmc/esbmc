def f():
    return 1


def h():
    return "a"


fs = [f, h]
fs[0] = h
assert fs[0]() == 1
