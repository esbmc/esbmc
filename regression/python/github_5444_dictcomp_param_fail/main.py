def f(g):
    d = {v: 0 for u, v in g}
    return len(d)


assert f({('A', 'B'): 3, ('A', 'C'): 5}) == 3
