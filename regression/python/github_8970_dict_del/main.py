# A deleted key is gone from the dict and the others keep their function.
def f():
    return "a"


def g():
    return 1


d = {"a": f, "b": g}
c = nondet_bool()
if c:
    del d["b"]
r = d["a"]()
assert r == "a"
assert ("b" in d) == (not c)
