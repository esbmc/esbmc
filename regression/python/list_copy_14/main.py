def test() -> None:
    arr: list[int] = [1, 2, 3]
    copied: list[int] = arr.copy()
    
    assert copied[-1] == 3
    assert copied[-2] == 2
    assert copied[-3] == 1

test()
