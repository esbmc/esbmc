def main() -> None:
    # list.insert(i, x) with a negative i counts from the end and clamps to
    # [0, len] (CPython never raises for insert). The model treated the index
    # as unsigned, so a negative index wrapped to a huge value and appended.
    a = [1, 2, 3]
    a.insert(-1, 9)
    assert a == [1, 2, 9, 3]

    b = [1, 2, 3]
    b.insert(-2, 9)
    assert b == [1, 9, 2, 3]

    # -len inserts at the front; a too-negative index clamps to the front.
    c = [1, 2, 3]
    c.insert(-3, 9)
    assert c == [9, 1, 2, 3]

    d = [1, 2, 3]
    d.insert(-5, 9)
    assert d == [9, 1, 2, 3]

    # The non-negative forms are unchanged (front, middle, beyond-end append).
    e = [1, 2, 3]
    e.insert(0, 9)
    assert e == [9, 1, 2, 3]

    f = [1, 2, 3]
    f.insert(5, 9)
    assert f == [1, 2, 3, 9]


main()
