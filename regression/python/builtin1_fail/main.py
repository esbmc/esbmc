def test_abs(x: int) -> int:
    result = abs(x)
    length = len([1, 2, 3, 10])
    return result + length

assert test_abs(-5) == 8  # should fail
