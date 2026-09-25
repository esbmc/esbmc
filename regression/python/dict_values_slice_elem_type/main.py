def run():
    w = {"a": (1, 2), "b": (3, 4)}
    vs = list(w.values())[:]
    u, v = vs[0]
    assert u == 1
    assert v == 2


run()
