def test() -> None:
    arr: list[int] = [1, 2, 3]
    copy1: list[int] = arr.copy()
    copy2: list[int] = arr.copy()
    
    copy1[0] = 10
    copy2[0] = 20
    
    assert arr[0] == 1
    assert copy1[0] == 10
    assert copy2[0] == 20

test()
