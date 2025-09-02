def test_lambda() -> None:
    # Basic lambda with arithmetic
    add_ten: int = lambda x: x + 10
    assert add_ten(5) == 15
    
    # Lambda with multiple parameters
    multiply: int = lambda a, b, c: a * b * c
    assert multiply(2, 3, 4) == 24
    
    # Lambda with boolean logic
    is_positive: bool = lambda x: x > 0
    assert is_positive(5) == True
    assert is_positive(-3) == False

test_lambda()
