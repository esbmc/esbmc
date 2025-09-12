def test_valid_positive():
    """Valid positive integer string"""
    x = int("123")
    assert x == 123

def test_valid_negative():
    """Valid negative integer string"""
    x = int("-456")
    assert x == -456

def test_valid_explicit_positive():
    """Valid positive integer with explicit + sign"""
    x = int("+789")
    assert x == 789

def test_valid_zero():
    """Valid zero string"""
    x = int("0")
    assert x == 0

def test_valid_negative_zero():
    """Valid negative zero (becomes 0)"""
    x = int("-0")
    assert x == 0

def test_valid_large_number():
    """Valid large integer within int range"""
    x = int("2147483647")  # Max 32-bit signed int
    assert x == 2147483647

def test_valid_multiple_digits():
    """Valid multi-digit numbers"""
    a = int("1")
    b = int("12")
    c = int("123")
    d = int("1234")
    assert a == 1
    assert b == 12
    assert c == 123
    assert d == 1234

def test_valid_in_expressions():
    """Valid int() calls used in expressions"""
    result = int("10") + int("20")
    assert result == 30

def test_valid_sequential():
    """Multiple valid conversions in sequence"""
    a = int("42")
    b = int("-17")
    c = int("+99")
    total = a + b + c
    assert total == 124

# Main execution - all these should verify successfully
if __name__ == "__main__":
    test_valid_positive()
    test_valid_negative() 
    test_valid_explicit_positive()
    test_valid_zero()
    test_valid_negative_zero()
    test_valid_large_number()
    test_valid_multiple_digits()
    test_valid_in_expressions()
    test_valid_sequential()

