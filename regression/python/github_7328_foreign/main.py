def f(n: int):
    c = [(1, 2)]
    k = lambda t: t[0]

    def g():
        d = [(1.5, 2.5)]
        return k(d[0])

    assert k(c[0]) == 1


f(3)
