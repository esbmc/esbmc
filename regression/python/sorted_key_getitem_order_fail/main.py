def run():
    w = {(1, 2): 10, (3, 4): 20, (5, 6): 5}
    ks = sorted(w, key=w.__getitem__)
    # (1, 2) is ks[1], not ks[0]: the key must really order by the dict value.
    a, b = ks[0]
    assert a == 1


run()
