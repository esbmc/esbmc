# Every call of f1 passes f2 for a, so the call through a is a call of f2.
def f3():
    return "s"


def f4():
    return nondet_int()


def f2(a):
    return a()


def f1(a, b):
    return a(b)


u = nondet_bool()
d = f1(f2, f3 if u else f4)

if u:
    assert d == "s"
else:
    assert isinstance(d, int)
