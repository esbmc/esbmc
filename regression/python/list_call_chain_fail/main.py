def test() -> None:
    x = [1, 2, 3]
    x.copy().append(99)
    assert len(x) == 4


test()
