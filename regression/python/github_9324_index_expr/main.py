# The index of the call is i % 2, so the call goes to f or g by parity.
def f():
    return 1


def g():
    return "a"


fs = [f, g]
i = nondet_int()
r = fs[i % 2]()
if i % 2 == 0:
    assert r == 1
else:
    assert r == "a"
