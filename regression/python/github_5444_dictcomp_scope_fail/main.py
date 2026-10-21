def ends(g):
    v = 5
    d = {v: 0 for u, v in g}
    assert v == 2
    return d


ends({(1, 2): 3})
