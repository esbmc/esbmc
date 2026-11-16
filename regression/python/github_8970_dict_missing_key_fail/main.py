# The key "z" is not in d, so d[k] raises KeyError.
def f():
    return "a"


def g():
    return 1


d = {"a": f, "b": g}
c = nondet_bool()
k = "a" if c else "z"
r = d[k]()
