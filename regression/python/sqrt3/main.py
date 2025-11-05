def test_sqrt_small_float():
    """Test: sqrt of small positive float"""
    import math
    x: float = 1e-12
    result: float = math.sqrt(x)
    assert abs(result - 1e-6) < 1e-12

test_sqrt_small_float()
