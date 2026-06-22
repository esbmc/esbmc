def total(g):
    s = 0
    for k, w in g.items():
        s += w
    return s


assert total({'a': 3, 'b': 7}) == 10
