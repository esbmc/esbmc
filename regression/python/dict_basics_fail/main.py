def test() -> None:
    # pop with default on missing key returns the default value.
    d = {-1: 2}
    assert d.pop(7, 8) == 7


test()
