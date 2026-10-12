# A global the module binds twice, to values of different types, has no
# single type to read it at before either binding runs (#8369).
def f() -> int:
    return d["a"]


d = {"a": 1}
assert f() == 1
d = "a"
