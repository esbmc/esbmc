def run():
    w = {(1, 2): 10, (3, 4): 20, (5, 6): 5}
    first = 0
    for edge in sorted(w, key=w.__getitem__):
        u, v = edge
        if first == 0:
            # (1, 2) is what a key-ignoring sort puts first.
            assert u == 1
        first = first + 1


run()
