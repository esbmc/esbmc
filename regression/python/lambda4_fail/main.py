def test() -> None:
    add = lambda a, b: a + b
    assert add(2, 3) != 5
    mul = lambda a, b, c: a * b * c
    assert mul(2, 3, 4) != 24


test()
