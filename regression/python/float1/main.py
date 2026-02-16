def test_integer_zero():
    """Test converting integer zero"""
    x = float(0)
    assert x == 0.0

test_integer_zero()

def test_integer_positive():
    """Test converting positive integer"""
    x = float(42)
    assert x == 42.0

test_integer_positive()

def test_integer_negative():
    """Test converting negative integer"""
    x = float(-123)
    assert x == -123.0

test_integer_negative()

def test_integer_large():
    """Test converting large integer"""
    x = float(999999999)
    assert x == 999999999.0

test_integer_large()
