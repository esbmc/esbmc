def f3():
    return "x"


def f2(a):
    return a()


def f1(a, b):
    return a(b)


if nondet_bool():
    b = f3
else:
    b = f2
c = f1(f2, b)
assert c == "x"
