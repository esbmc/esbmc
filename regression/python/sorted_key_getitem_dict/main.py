def run():
    w = {(1, 2): 10, (3, 4): 20, (5, 6): 5}
    ks = sorted(w, key=w.__getitem__)
    # sorted by value: (5, 6) has 5, (1, 2) has 10, (3, 4) has 20.
    a, b = ks[0]
    assert a == 5
    assert b == 6


run()
