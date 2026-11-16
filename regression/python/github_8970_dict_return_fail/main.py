# As github_8970_dict_return, but the int path asserts what is false.
def f():
    return "a"


def g():
    return 1


def make():
    d = {"x": f, "y": g}
    return d


b = make()
key = "x" if nondet_bool() else "y"
r = b[key]()
if key == "y":
    assert r == "a"
