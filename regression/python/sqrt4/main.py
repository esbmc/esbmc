def test_sqrt_zero_variants():
    """Test: sqrt(0.0) and sqrt(-0.0)"""
    import math
    pos_zero = math.sqrt(0.0)
    neg_zero = math.sqrt(-0.0)
    assert pos_zero == 0.0
    assert neg_zero == 0.0


test_sqrt_zero_variants()
