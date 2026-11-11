# make reads d before it stores it: UnboundLocalError in CPython.
def f():
    return "a"


def g():
    return 1


def make():
    e = d
    d = {"x": f, "y": g}
    return d


b = make()
key = "x" if nondet_bool() else "y"
r = b[key]()
