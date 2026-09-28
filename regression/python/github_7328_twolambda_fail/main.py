def f(n: int):
    a = [(1, 2)]
    b = [(3.5, 4.5)]
    p = lambda t: t[0]
    q = lambda u: u[0]
    assert p(a[0]) == 1
    assert q(b[0]) == 4.5


f(3)
