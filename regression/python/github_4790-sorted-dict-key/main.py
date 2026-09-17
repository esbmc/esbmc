def run():
    w = {(1, 2): 20, (3, 4): 10}
    ks = sorted(w, key=w.__getitem__)
    u, v = ks[0]
    assert u == 3
    assert v == 4


run()
