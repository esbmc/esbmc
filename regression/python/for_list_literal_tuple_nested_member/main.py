def run():
    first = 0
    for item in [((1, 2), 3), ((4, 5), 6)]:
        a, b = item
        if first == 0:
            assert b == 3
        first = first + 1
    assert first == 2


run()
