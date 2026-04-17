import math

def test_comb_type_errors():
    """Test TypeError cases"""
    # Test with float arguments
    try:
        result = math.comb(5.5, 2)
        assert False, "Expected TypeError for float n"
    except TypeError:
        pass  # Expected
    
    try:
        result = math.comb(5, 2.5)
        assert False, "Expected TypeError for float k"
    except TypeError:
        pass  # Expected
    
    # Test with string arguments
    try:
        result = math.comb("5", 2)
        assert False, "Expected TypeError for string n"
    except TypeError:
        pass  # Expected
    
    try:
        result = math.comb(5, "2")
        assert False, "Expected TypeError for string k"
    except TypeError:
        pass  # Expected
    
    # Test with None arguments
    try:
        result = math.comb(None, 2)
        assert False, "Expected TypeError for None n"
    except TypeError:
        pass  # Expected
    
    try:
        result = math.comb(5, None)
        assert False, "Expected TypeError for None k"
    except TypeError:
        pass  # Expected

def test_comb_value_errors():
    """Test ValueError cases"""
    # Test with negative n
    try:
        result = math.comb(-5, 2)
        assert False, "Expected ValueError for negative n"
    except ValueError:
        pass  # Expected
    
    # Test with negative k
    try:
        result = math.comb(5, -2)
        assert False, "Expected ValueError for negative k"
    except ValueError:
        pass  # Expected
    
    # Test with both negative
    try:
        result = math.comb(-5, -2)
        assert False, "Expected ValueError for both negative"
    except ValueError:
        pass  # Expected

def test_comb_valid_cases():
    """Test that valid inputs still work correctly"""
    # Basic cases
    assert math.comb(5, 2) == 10
    assert math.comb(10, 3) == 120
    
    # Edge cases
    assert math.comb(5, 0) == 1
    assert math.comb(5, 5) == 1
    assert math.comb(5, 1) == 5
    assert math.comb(5, 4) == 5
    
    # k > n should return 0
    assert math.comb(3, 5) == 0
    
    # Symmetry property
    assert math.comb(7, 3) == math.comb(7, 4)

test_comb_type_errors()
test_comb_value_errors()
test_comb_valid_cases()
