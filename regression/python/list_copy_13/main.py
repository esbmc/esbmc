def test() -> None:
    original: list[int] = [1, 2, 3]
    copied: list[int] = original.copy()
    
    original.clear()
    
    assert len(original) == 0
    assert len(copied) == 3
    assert copied[0] == 1

test()
