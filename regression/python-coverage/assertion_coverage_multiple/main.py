def process(n: int) -> int:
    assert n >= 0, "n must be non-negative"
    if n > 10:
        assert n < 100, "n must be less than 100"
        return n * 2
    return n

# Cover all assertions
process(5)
process(50)
