def test() -> None:
    arr: list[int] = []
    copied: list[int] = arr.copy()
    
    assert len(copied) == 0

test()
