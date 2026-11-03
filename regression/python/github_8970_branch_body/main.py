# The call through h is in an if body, so the rest of the module is copied
# into both branches and each path sees its own result type.
def f():
    return "a"


def g():
    return 1


use = nondet_bool()
h = f if nondet_bool() else g
if use:
    r = h()
    if h is f:
        assert type(r) == str
        assert r.upper() == "A"
    else:
        assert r + 1 == 2
    mark = 1
else:
    mark = 2
assert mark == (1 if use else 2)
