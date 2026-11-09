def f(x):
    [a, b], c = x
    return a


assert f(([1, 2, 3], 4)) == 1
