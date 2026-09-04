def run():
    first = 0
    # A str member is not a type the list element read can size, so the
    # literal keeps no tuple annotation and the unpack is refused.
    for pair in [("x", "y"), ("p", "q")]:
        a, b = pair
        if first == 0:
            assert a == "x"
            assert b == "y"
        first = first + 1
    assert first == 2


run()
