def run():
    first = 0
    # A tuple member would annotate the element as tuple[tuple, int], whose
    # bare inner tuple is the #5444 erosion; the unpack is refused instead.
    for item in [((1, 2), 3), ((4, 5), 6)]:
        a, b = item
        if first == 0:
            assert b == 3
        first = first + 1
    assert first == 2


run()
