def test_sqrt_conditional_input():
    """Test: sqrt of conditional expression"""
    import math
    x = 9 if True else 4
    result = math.sqrt(x)
    assert result == 3.0

test_sqrt_conditional_input()
