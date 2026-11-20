def k(c, items):
    w, v = items[0]
    return w + c


def t(d, e):
    assert k(*d) == e


t([1, [[60, 10]]], 62)
