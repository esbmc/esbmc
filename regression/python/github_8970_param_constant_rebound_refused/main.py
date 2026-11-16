# The parameter is assigned in the body, so it does not stay the function that was passed.
def f3():
    return "s"


def f4():
    return nondet_int()


def f2(a):
    return a()


def f1(a, b):
    a = f2
    return a(b)


u = nondet_bool()
d = f1(f2, f3 if u else f4)
