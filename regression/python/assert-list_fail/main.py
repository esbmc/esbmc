def test_index(lst: list[int], idx: int):
    assert 0 <= idx < len(lst)
    return lst[idx]

test_index([1,2,3], 3)
