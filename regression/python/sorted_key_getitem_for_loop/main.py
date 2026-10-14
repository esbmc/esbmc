def run():
    w = {(1, 2): 10, (3, 4): 20, (5, 6): 5}
    first = 0
    for edge in sorted(w, key=w.__getitem__):
        u, v = edge
        if first == 0:
            # sorted by value puts (5, 6) first, not (1, 2).
            assert u == 5
            assert v == 6
        first = first + 1
    assert first == 3


run()
