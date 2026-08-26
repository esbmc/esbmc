def grow(d):
    d[(5, 6)] = 5


def run():
    w = {(1, 2): 20, (3, 4): 10}
    grow(w)
    ks = sorted(w, key=w.__getitem__)
    u, v = ks[0]
    assert u == 5


run()
