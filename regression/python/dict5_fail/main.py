def test_comparisons():
    limits: dict = {'min': 0, 'max': 100, 'default': 50}
    
    x: int = limits['default']
    assert x < limits['min']
    assert x > limits['max']
    assert limits['min'] > limits['max']

test_comparisons()
