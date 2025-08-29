def test() -> None:
    add:int = lambda a, b: a + b
    assert add(2, 3) == 6  # wrong on purpose

test()

