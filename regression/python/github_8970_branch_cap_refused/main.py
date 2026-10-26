# Two calls in sequential ifs need more than eight copies of the end.
def f():
    return "a"


def g():
    return 1


def k():
    return 2.5


h = f if nondet_bool() else (g if nondet_bool() else k)
if nondet_bool():
    r = h()
if nondet_bool():
    s = h()
if nondet_bool():
    t = h()
