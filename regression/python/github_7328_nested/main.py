def f(n: int):
    c = [(1, 2)]
    k = lambda t: t[0]

    def g():
        c = [(1.5, 2.5)]
        return k(c[0])

    assert g() == 1.5


f(3)
