from esbmc import nondet_int, __ESBMC_assume

def absolute_value(x: int) -> int:
    """Compute absolute value."""
    if x < 0:
        return -x
    return x


def test_absolute_value() -> None:
    """Verify absolute value properties."""
    x: int = nondet_int()
    __ESBMC_assume(x > -1000 and x < 1000)  # Avoid overflow

    result = absolute_value(x)

    # Property: result is always non-negative
    assert result >= 0, "Absolute value is non-negative"

    # Property: result equals x or -x
    assert result == x or result == -x, "Result is x or -x"

test_absolute_value()
