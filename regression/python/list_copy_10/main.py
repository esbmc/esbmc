def make_copy(original: list[int]) -> list[int]:
    return original.copy()

def test() -> None:
    arr: list[int] = [5, 10, 15]
    result: list[int] = make_copy(arr)
    
    assert len(result) == 3
    assert result[0] == 5
    assert result[1] == 10
    assert result[2] == 15

test()
