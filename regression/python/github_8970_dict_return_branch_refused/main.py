# make branches, so its body is not straight-line.
def f():
    return "a"


def g():
    return 1


def make():
    d = {"x": f, "y": g}
    if nondet_bool():
        d = {"x": g, "y": f}
    return d


b = make()
key = "x" if nondet_bool() else "y"
r = b[key]()
