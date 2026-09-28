def f(x: int):
    c = [(1, 2), (3, 4)]
    k = lambda x: x[0]
    assert k(c[1]) == 3


f(3)
