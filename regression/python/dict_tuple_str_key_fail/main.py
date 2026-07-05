def test() -> None:
    d = {("x", "y"): 1, ("x", "z"): 2}
    assert d[("x", "y")] == 9


test()
