# Issue #8920: a starred list longer than f's parameters raises TypeError
# even when f has defaults.
def f(a, b=1):
    return a + b


def test(args, expected):
    assert f(*args) == expected


test([1, 2, 3], 3)
