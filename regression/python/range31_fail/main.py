def helper(lst: list[int]) -> int:
    return lst[0]

def test_range_as_argument():
    l = list(range(5))
    result = helper(l)
    assert result == 1

test_range_as_argument()
