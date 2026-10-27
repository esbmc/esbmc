# A named dict passed as an argument and changed through an alias (#9211).
def f():
    return "a"


def g():
    return 1


def run(table):
    return table["a"]()


d = {"a": f}
e = d
e["a"] = g
assert run(d) != 1
