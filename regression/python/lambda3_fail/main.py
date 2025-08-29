def test() -> None:
    x:float = lambda a : a + 1.1
    assert x(1.9) < 3.0
test()
