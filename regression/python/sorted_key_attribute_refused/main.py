def run():
    w = {(1, 2): 10, (3, 4): 20}
    # `for x in sorted(...)` builds its iterable assignment after the pass that
    # lowers a key'd call, so the scan never sees this one and it must still be
    # refused rather than sorted with the key ignored. The direct-assign form is
    # supported -- see sorted_key_getitem_dict.
    for edge in sorted(w, key=w.__getitem__):
        u, v = edge
        assert u < v


run()
