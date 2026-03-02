def test_abs(x: int) -> int:
    result = abs(x)
    length = len([1, 2, 3])
    return result + length


assert test_abs(-5) == 8  # abs(-5) + len([1,2,3]) = 5 + 3 = 8
