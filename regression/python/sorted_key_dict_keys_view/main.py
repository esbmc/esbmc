def run():
    w = {1: 20, 2: 10, 3: 30}
    w.get(1)
    ks = sorted(w.keys(), key=w.__getitem__)
    assert ks[0] == 2
    assert ks[2] == 3


run()
