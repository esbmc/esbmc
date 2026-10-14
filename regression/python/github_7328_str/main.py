def f(n: int):
    c = [(1, 2)]
    k = lambda t: t[0]
    assert k(c[0]) == 1
    assert k("ab") == "a"


f(3)
