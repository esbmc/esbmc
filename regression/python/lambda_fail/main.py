def test() -> None:
    x:int = lambda a : a + 10
    assert x(5) == 16

test()
