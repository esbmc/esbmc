def test() -> None:
    arr: list[str] = ["hello", "world"]
    copied: list[str] = arr.copy()

    assert len(copied) == 2
    assert copied[0] == "hello"
    assert copied[1] == "world"


test()
