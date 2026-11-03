# The callee is a conditional expression of two functions.
def f(x):
    return "a"


def g(x):
    return 1


c = nondet_bool()
r = (f if c else g)(0)
assert isinstance(r, str) == c
if c:
    assert r == "a"
else:
    assert r == 1
