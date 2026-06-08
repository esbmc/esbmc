def test() -> None:
    a = {}
    a.setdefault(1, []).append(1.0)
    a.setdefault(1, []).append(2.0)
    assert len(a.setdefault(1, [])) == 2
    a.setdefault(2, []).append(9.0)
    assert len(a.setdefault(2, [])) == 1


test()
