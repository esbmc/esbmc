def test() -> None:
    arr: list[int] = [1, 2, 3]
    copy1: list[int] = arr.copy()
    copy2: list[int] = copy1.copy()
    
    copy2[0] = 99
    
    assert arr[0] == 1
    assert copy1[0] == 1
    assert copy2[0] == 99

test()
