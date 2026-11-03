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
    assert d == "t"
else:
    assert isinstance(d, int)
