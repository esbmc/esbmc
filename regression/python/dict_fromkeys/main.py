def test() -> None:
    d = dict.fromkeys([1, 2, 3], 7)
    assert len(d) == 3
    assert d[1] == 7
    assert d[2] == 7
    assert d[3] == 7

    f = dict.fromkeys([1, 2, 3], 4.0)
    assert f[1] == 4.0

    n = dict.fromkeys([1, 2, 3])
    assert len(n) == 3

    # Duplicate keys collapse per Python semantics.
    dup = dict.fromkeys([1, 1, 2, 3, 3], 5)
    assert len(dup) == 3


test()
