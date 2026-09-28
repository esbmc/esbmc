def f(n: int):
    c = [(1, 2)] + [(3.5, 4.5)]
    k = lambda t: t[0]
    assert k(c[1]) == 3.5


f(3)
