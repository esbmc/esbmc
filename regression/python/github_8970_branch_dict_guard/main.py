# The lookup is guarded by a membership test and the call is in the if body.
def f():
    return "a"


def g():
    return 1


d = {"x": f, "y": g}
k = "x" if nondet_bool() else ("y" if nondet_bool() else "z")
if k in d:
    r = d[k]()
    assert (r == "a") == (k == "x")
else:
    assert k == "z"
