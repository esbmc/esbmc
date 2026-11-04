def run():
    first = 0
    for pair in [("x", "y"), ("p", "q")]:
        a, b = pair
        if first == 0:
            assert a == "x"
            assert b == "q"
        first = first + 1
    assert first == 2


run()
