def test() -> None:
    arr: list[bool] = [True, False, True]
    copied: list[bool] = arr.copy()
    
    assert len(copied) == 3
    assert copied[0] == True
    assert copied[1] == False
    assert copied[2] == True

test()
