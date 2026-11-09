def f(x):
    [a, b], c = x
    return a + b + c


assert f(([1, 2], 3)) == 6
assert f(([4, 5], 6)) == 14
