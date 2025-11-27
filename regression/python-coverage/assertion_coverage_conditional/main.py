def check_bounds(n: int) -> int:
    assert n >= 0, "n must be non-negative"
    if n > 10:
        assert n < 100, "n must be less than 100"
        return 1
    return 0

# Only cover first assertion
check_bounds(5)
