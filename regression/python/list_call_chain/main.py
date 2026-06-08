def test() -> None:
    x = [1, 2, 3]
    x.copy().append(99)
    assert len(x) == 3
    assert x[0] == 1
    assert x[2] == 3


test()
