# d["a"] never takes the 5, but the dict then holds a value that is no function.
def f():
    return "a"


def g():
    return 1


d = {"a": f, "b": g}
assert d["a"]() == "a"
if False:
    d["a"] = 5
