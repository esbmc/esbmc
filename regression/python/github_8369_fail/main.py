# The global read through the forward reference is checked (#8369).
def f() -> int:
    return d["a"]


d = {"a": 1}
assert f() == 2
