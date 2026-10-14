def main() -> None:
    # list() over a tuple literal.
    a = list((2, 3))
    assert a[0] == 2 and a[1] == 3 and len(a) == 2

    # list() over a tuple held in a variable.
    t = (8, 9)
    b = list(t)
    assert b[0] == 8 and b[1] == 9 and len(b) == 2

    # A single-element tuple.
    c = list((5,))
    assert c[0] == 5 and len(c) == 1

    # The result is a real, mutable list.
    d = list((1, 2))
    d.append(3)
    assert d[2] == 3 and len(d) == 3

    # An empty tuple yields an empty list.
    f = list(())
    assert len(f) == 0

    # list() over a list / range is unaffected.
    e = list([4, 5])
    assert e[1] == 5


main()
