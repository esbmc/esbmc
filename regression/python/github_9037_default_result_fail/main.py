# f2 returns what its function default h returns, an int like g (#9037).
def g():
    return 1


def h():
    return 2


def f2(a=h):
    return a()


def f1(a):
    return a()


if nondet_bool():
    c = f1(g)
    assert c == 1
else:
    c = f1(f2)
    assert c == 1
