def f():
    return "a"


def g():
    return 1


a, *b = f, g, f
assert a() == "a"
assert b[0]() == 1
assert b[1]() == "a"
