def test_sqrt_negative_float():
    """Test: sqrt(-0.5) should trigger domain error"""
    import math
    try:
        math.sqrt(-0.5)
        assert False, "Expected ValueError for negative float input"
    except ValueError:
        pass


test_sqrt_negative_float()
