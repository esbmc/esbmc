def f(n: int):
    c = [(1, 2), (3, 4)]
    k = lambda t: t[0]
    assert k(c[7]) == 1


f(3)
