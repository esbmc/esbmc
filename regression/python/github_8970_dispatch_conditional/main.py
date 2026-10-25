# h may be f or g, which return str and int. Each path sees its own result
# type, so type(), str methods and the function's identity agree with CPython.
def f():
    return "a"


def g():
    return 1


c = nondet_bool()
h = f if c else g
assert callable(h)
r = h()
if h is f:
    assert type(r) == str
    assert r.upper() == "A"
else:
    assert isinstance(r, int)
    assert r + 1 == 2
assert (h is not g) == c
