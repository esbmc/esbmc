def test() -> None:
    x = {"a": 1}.get("missing", 3)
    assert x == 3

    y = {"a": 1}.get("a", 99)
    assert y == 1

    z = list({}.items())
    assert len(z) == 0

test()
