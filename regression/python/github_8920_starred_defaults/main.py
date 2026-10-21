# Issue #8920: f(*xs) on a parameter xs when f has defaults among the
# starred slots. Each length from the required count up to the parameter
# count binds, and the defaults fill the rest.
def f(a, b=1):
    return a + b


def g(a, b=10, c=100):
    return a + b + c


def test(args, expected):
    assert f(*args) == expected


def test_g(rest, expected):
    assert g(1, *rest) == expected


def too_many(args):
    caught = False
    try:
        f(*args)
    except TypeError:
        caught = True
    return caught


test([1, 2], 3)
test([5], 6)
test_g([], 111)
test_g([2], 103)
test_g([2, 3], 6)
assert too_many([1, 2, 3])
