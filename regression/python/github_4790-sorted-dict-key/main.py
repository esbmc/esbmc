def run():
    w = {(1, 2): 10, (3, 4): 20}
    ks = sorted(w, key=w.__getitem__)
    u, v = ks[0]
    assert u == 1


run()
