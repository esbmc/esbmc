# make stores "Y", so b["y"] is a KeyError.
def f():
    return "a"


def g():
    return 1


def make():
    d = {"x": f, "Y": g}
    return d


b = make()
key = "x" if nondet_bool() else "y"
r = b[key]()
