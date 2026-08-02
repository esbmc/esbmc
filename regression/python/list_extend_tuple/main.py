def main() -> None:
    # extend() accepts any iterable, including a tuple literal.
    a = [1]
    a.extend((2, 3))
    assert a[1] == 2 and a[2] == 3 and len(a) == 3

    # A single-element tuple.
    b = [5]
    b.extend((7,))
    assert b[1] == 7 and len(b) == 2

    # A tuple held in a variable works too.
    t = (8, 9)
    c = [0]
    c.extend(t)
    assert c[2] == 9 and len(c) == 3

    # Extending with a list is unaffected.
    d = [1]
    d.extend([2, 3])
    assert d[2] == 3

    # A mixed-type tuple keeps the right length.
    e = [0]
    e.extend((1, "x"))
    assert len(e) == 3


main()
