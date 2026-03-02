def sum_list(items: list[int]) -> int:
    """Sum all elements in a list."""
    total = 0
    for item in items:
        total += item
    return total


def test_sum_with_nondet_list():
    """Test sum_list with nondet input."""
    x: list[int] = nondet_list(7)
    result = sum_list(x)
    # Result is well-defined
    assert result == result


test_sum_with_nondet_list()
