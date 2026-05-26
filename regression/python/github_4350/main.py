def main() -> None:
    t = (1, 2, 3, 4, 5)

    # Basic slice [start:stop].
    s = t[1:3]
    assert s[0] == 2
    assert s[1] == 3

    # Open ends.
    a = t[:2]
    assert a[0] == 1 and a[1] == 2
    b = t[3:]
    assert b[0] == 4 and b[1] == 5

    # Negative bounds.
    c = t[-2:]
    assert c[0] == 4 and c[1] == 5

    # Step.
    d = t[::2]
    assert d[0] == 1 and d[1] == 3 and d[2] == 5

    # Reverse via step=-1.
    e = t[::-1]
    assert e[0] == 5 and e[4] == 1
main()
