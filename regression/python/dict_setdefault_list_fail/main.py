def test() -> None:
    a = {}
    a.setdefault(1, []).append(1.0)
    assert len(a.setdefault(1, [])) == 0


test()
