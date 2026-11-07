def f(x):
    [a, b], c = x
    return a + b + c


def g(x):
    [a, b], c = x
    assert a == "s" and b == "t"


assert f(([1, 2], 3)) == 6
assert f(([4, 5], 6)) == 15
g((["s", "t"], 3))
