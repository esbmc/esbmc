# A tuple() argument is a list marked as a tuple, so isinstance(t, tuple)
# holds in the callee as it does in CPython (#8777).


def f(t):
    assert isinstance(t, tuple)
    a, b, c = t
    return a + b + c


assert f(tuple(range(3))) == 3
