def test() -> None:
    arr: list[int] = [1, 2, 3]
    copied: list[int] = arr.copy()
    
    assert len(copied) == 3
    assert copied[0] == 1
    assert copied[1] == 2
    assert copied[2] == 3

test()
