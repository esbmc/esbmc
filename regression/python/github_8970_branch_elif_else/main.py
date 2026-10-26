# The calls are in an elif body and an else body, and the first branch has none.
def f():
    return "a"


def g():
    return 1


first = nondet_bool()
second = nondet_bool()
h = f if nondet_bool() else g
if first:
    tag = 1
elif second:
    r = h()
    assert (r == "a") == (h is f)
    tag = 2
else:
    r = h()
    assert (r == 1) == (h is g)
    tag = 3
assert (tag == 1) == first
assert (tag == 2) == (second and not first)
