def f(n: int):
    c = [(1, 2)] + [(3, 4)]
    k = lambda t: t[0]
    assert k(c[1]) == 3


f(3)
