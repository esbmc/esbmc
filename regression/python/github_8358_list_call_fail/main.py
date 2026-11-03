def f(t):
    a, b, c = t
    return a + b + c


assert f(list(range(3))) == 2
