# A tuple() argument leaves the parameter a bare tuple even when another call
# site passes a literal, and the unpacking refuses it.


def f(t):
    a, b, c = t
    return a + b + c


assert f(tuple(range(3))) == 3
assert f((1, 2, 3)) == 6
