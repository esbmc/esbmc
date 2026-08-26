def run():
    w = {(1, 2): 20, (3, 4): 10}
    w2 = w
    w2[(5, 6)] = 5
    ks = sorted(w, key=w.__getitem__)
    u, v = ks[0]
    assert u == 3


run()
