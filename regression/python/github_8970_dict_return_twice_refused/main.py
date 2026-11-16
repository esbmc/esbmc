# make is called twice, so its body is not inlined.
def f():
    return "a"


def g():
    return 1


def make():
    d = {"x": f, "y": g}
    return d


b = make()
c = make()
key = "x" if nondet_bool() else "y"
r = b[key]()
