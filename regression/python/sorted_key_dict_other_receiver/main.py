def run():
    w = {1: 20, 2: 10}
    v = {1: 1, 2: 2}
    ks = sorted(w, key=v.__getitem__)
    assert ks[0] == 1


run()
