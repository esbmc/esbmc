def test_lambda_expressions():
    # Basic arithmetic lambda
    add_ten = lambda x: x + 10
    result1 = add_ten(5)
    assert result1 == 15

    # Multi-parameter lambda
    calculate_volume = lambda length, width, height: length * width * height
    volume = calculate_volume(2.0, 3.0, 4.0)
    assert volume == 24.0

    # Lambda with conditional logic
    absolute_diff = lambda a, b: a - b if a > b else b - a
    diff1 = absolute_diff(10, 3)
    diff2 = absolute_diff(3, 10)
    assert diff1 == 7
    assert diff2 == 7

    # Lambda for boolean operations
    is_in_range = lambda x, lower, upper: lower <= x <= upper
    assert is_in_range(5, 1, 10) == True
    assert is_in_range(15, 1, 10) == False


test_lambda_expressions()
