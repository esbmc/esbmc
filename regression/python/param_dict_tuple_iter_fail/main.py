def f(edges):
    s = 0
    for u, v in edges:
        s += u * 10 + v
    _ = edges.items()
    return s


assert f({(1, 2): 0, (3, 4): 0}) == 45
