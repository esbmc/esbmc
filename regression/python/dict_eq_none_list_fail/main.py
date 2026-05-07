def test() -> None:
    a = {}
    a.setdefault(1, []).append(1.0)
    assert a == {1: [2.0]}

    assert dict.fromkeys([1, 2, 3]) == {1: None, 2: None, 3: None}


test()
