# Calling the function before the module binds the global raises
# NameError in CPython (#8369).
def f() -> int:
    return d["a"]


assert f() == 1
d = {"a": 1}
