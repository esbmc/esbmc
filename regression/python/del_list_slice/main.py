def main() -> None:
    # del a[lower:upper] removes the slice. It was desugared to a.pop(slice),
    # which passed the Slice node as a pop index (invalid → internal error). It
    # now routes through slice assignment with an empty replacement.
    a = [1, 2, 3, 4]
    del a[1:3]
    assert a == [1, 4]

    b = [1, 2, 3, 4]
    del b[:2]
    assert b == [3, 4]

    c = [1, 2, 3, 4]
    del c[2:]
    assert c == [1, 2]

    d = [1, 2, 3, 4, 5]
    del d[1:3]
    assert len(d) == 3 and d[1] == 4

    # Single-index del still works.
    e = [1, 2, 3]
    del e[1]
    assert e == [1, 3]


main()
