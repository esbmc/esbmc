def run():
    w = {1: 20, 2: 10, 1: 5}
    ks = sorted(w, key=w.__getitem__)
    assert ks[0] == 1


run()
