# A function body reads a module global bound after the def (#8369).
def f() -> int:
    return d["a"]


d = {"a": 1}
assert f() == 1
