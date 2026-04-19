def test_no_arguments():
    """Test float() with no arguments returns 0.0"""
    x = float()
    assert x == 0.0
    assert isinstance(x, float)

test_no_arguments()
