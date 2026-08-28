def run():
    first = 0
    for edge in [(5, 6), (1, 2), (3, 4)]:
        u, v = edge
        if first == 0:
            assert u == 5
            assert v == 6
        first = first + 1
    assert first == 3


run()
