# f2 returns what its function default h returns, a str, while g returns an
# int, so the call through f1's parameter has no single type (#9037).
def g():
    return 1


def h():
    return "a"


def f2(a=h):
    return a()


def f1(a):
    return a()


if nondet_bool():
    c = f1(g)
    assert c == 1
else:
    c = f1(f2)
    assert c == "a"
