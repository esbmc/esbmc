def run():
    w = {(1, 2): 10, (3, 4): 20}
    # Only __getitem__ is emitted as an operator; any other bound method would
    # be emitted as a call that nothing gets the chance to rewrite, so it must
    # still be refused rather than sorted with the key ignored.
    for edge in sorted(w, key=w.get):
        u, v = edge
        assert u < v


run()
