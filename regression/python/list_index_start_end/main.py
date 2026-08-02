def main() -> None:
    # list.index(x, start[, end]) searches the slice l[start:end] and returns
    # the absolute index. The start/end arguments were previously unsupported
    # ("takes exactly one argument").
    a = [1, 2, 1, 2]
    assert a.index(2, 2) == 3

    b = [1, 2, 1, 2, 1]
    assert b.index(1, 1, 4) == 2

    # start == 0 matches the plain one-argument form.
    c = [5, 6, 7]
    assert c.index(6, 0) == 1

    # Negative start/end follow CPython slice-bound normalization.
    d = [1, 2, 3, 2]
    assert d.index(2, -2) == 3

    e = [1, 2, 3, 2, 5]
    assert e.index(2, 0, -1) == 1

    # The one-argument form is unchanged.
    f = [10, 20, 30]
    assert f.index(20) == 1


main()
