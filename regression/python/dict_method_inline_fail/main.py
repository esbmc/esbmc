def test() -> None:
    x = {"a": 1}.get("missing", 3)
    assert x == 99

test()
