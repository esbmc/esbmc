def test_string_int_compare(x: str) -> bool:
    return int(x) > 0 and int(x) < 100


def test_range_check(x: str) -> bool:
    val = int(x)
    return val >= 10 and val <= 20


assert test_string_int_compare("50")
assert test_range_check("15")
