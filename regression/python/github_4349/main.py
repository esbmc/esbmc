def main() -> None:
    # Concatenation of two int tuples extends the tuple type.
    a = (1, 2)
    b = (3, 4)
    c = a + b
    assert c[0] == 1
    assert c[2] == 3
    assert c[3] == 4

    # Concatenation with mixed-element types is also allowed; the result
    # tuple keeps each element's type.
    d = (10, 20)
    e = (30, 40, 50)
    f = d + e
    assert f[4] == 50

    # Repetition with a constant int builds an n-fold repeat.
    g = (7, 8)
    h = g * 3
    assert h[0] == 7
    assert h[5] == 8

    # Concat result is a fresh tuple; original a/b unchanged.
    assert a[0] == 1 and a[1] == 2
main()
