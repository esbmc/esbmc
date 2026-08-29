def f(n: int):
    c = [(1, 2), (3, 4)]
    i = 1
    k = lambda t: t[0]
    assert k(c[i]) == 3


f(3)
