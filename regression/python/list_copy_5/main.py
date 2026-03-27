def test() -> None:
    arr: list[float] = [1.5, 2.5, 3.5]
    copied: list[float] = arr.copy()
    
    assert len(copied) == 3
    assert copied[0] == 1.5
    assert copied[1] == 2.5
    assert copied[2] == 3.5

test()
