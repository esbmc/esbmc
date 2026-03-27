def test_range_check(x: str) -> bool:
    val = int(x)
    return val >= 10 and val <= 20


assert test_range_check("25")  # 25 is not in range [10, 20]
