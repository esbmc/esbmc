def test_leading_space(x: str) -> bool:
    return int(x) == 42


def test_trailing_space(x: str) -> bool:
    return int(x) == 42


def test_both_spaces(x: str) -> bool:
    return int(x) == 42


assert test_leading_space("  42")
assert test_trailing_space("42  ")
assert test_both_spaces("  42  ")
