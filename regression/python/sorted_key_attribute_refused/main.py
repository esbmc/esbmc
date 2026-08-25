def run():
    w = {(1, 2): 10, (3, 4): 20}
    # A bound-method key is not one of the shapes the scan lowering accepts, so
    # this must still be refused rather than sorted with the key ignored.
    for edge in sorted(w, key=w.__getitem__):
        u, v = edge
        assert u < v


run()
