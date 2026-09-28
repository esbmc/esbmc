def f(n: int):
    a = [(1, 2), (3, 4)]
    b = [(1.5, 2.5)]
    k = lambda t: t[0]
    assert k(a[0]) == 1
    assert k(b[0]) == 1.5


f(3)
