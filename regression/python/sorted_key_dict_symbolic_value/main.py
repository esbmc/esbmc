def run(a: int):
    w = {1: a, 2: 10}
    ks = sorted(w, key=w.__getitem__)
    assert ks[0] == 2


run(30)
