def run():
    w = {(1, 2): 10, (3, 4): 20}
    ks = list(w.keys())[:]
    u, v = ks[0]
    # (3, 4) is ks[1], not ks[0]: the slice must keep the elements in order,
    # not merely keep them unpackable.
    assert u == 3


run()
