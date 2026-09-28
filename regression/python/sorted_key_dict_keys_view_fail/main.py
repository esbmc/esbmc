def run():
    w = {1: 20, 2: 10, 3: 30}
    ks = sorted(w.keys(), key=w.__getitem__)
    assert ks[0] == 1


run()
