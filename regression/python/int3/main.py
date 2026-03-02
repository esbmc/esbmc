def test_no_arguments():
    """Test int() with no arguments returns 0"""
    x = int()
    assert x == 0
    assert isinstance(x, int)


test_no_arguments()
