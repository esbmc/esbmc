def f(n: int):
    a = [(1, 2), (3, 4)]
    b = [(5, 6)]
    k = lambda t: t[0]
    assert k(a[0]) == 1
    assert k(b[0]) == 6


f(3)
