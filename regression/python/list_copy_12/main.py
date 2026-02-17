def test() -> None:
    arr: list[int] = [42]
    copied: list[int] = arr.copy()
    
    assert len(copied) == 1
    assert copied[0] == 42

test()
