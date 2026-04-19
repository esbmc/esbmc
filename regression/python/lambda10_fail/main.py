def test_lambda_expressions():
    """Comprehensive lambda expression testing"""
    # Multi-parameter lambda
    calculate_volume = lambda length, width, height: length * width * height
    volume = calculate_volume(2.0, 3.0, 4.0)
    assert volume == 24.1

    # Lambda with conditional logic
    absolute_diff = lambda a, b: a - b if a > b else b - a
    diff1 = absolute_diff(10, 3)
    diff2 = absolute_diff(3, 10)
    assert diff1 == 8
    assert diff2 == 8

    # Lambda for boolean operations
    is_in_range = lambda x, lower, upper: lower <= x <= upper
    assert is_in_range(5, 1, 10) == False
    assert is_in_range(15, 1, 10) == True

test_lambda_expressions()
