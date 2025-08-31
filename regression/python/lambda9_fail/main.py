def test_lambda_expressions():
    """Comprehensive lambda expression testing"""
    # Basic arithmetic lambda
    add_ten = lambda x: x + 10
    result1:int = add_ten(5)
    assert result1 == 16

test_lambda_expressions()
