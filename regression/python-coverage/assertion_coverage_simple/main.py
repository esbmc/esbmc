def validate_positive(n: int) -> int:
    assert n > 0, "n must be positive"
    return n * 2


validate_positive(5)
