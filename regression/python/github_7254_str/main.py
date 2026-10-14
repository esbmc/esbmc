def less(a, b):
    assert a < b


def outer(x, ys):
    less(x, ys[0])


outer("abc", ["abd"])
