def run():
    first = 0
    for edge in [(5, 6), (1, 2), (3, 4)]:
        u, v = edge
        if first == 1:
            # (1, 2) is the second element; u is 1, not 5.
            assert u == 5
        first = first + 1


run()
