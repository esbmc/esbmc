# The lookup raises KeyError for "z" before the rest of the module runs.
def f():
    return "a"


def g():
    return 1


d = {"x": f, "y": g}
k = "x" if nondet_bool() else "z"
if nondet_bool():
    r = d[k]()
assert True
