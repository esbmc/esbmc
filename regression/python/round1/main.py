# Simple round() test with assert
def test_round():
    # Test basic round() without ndigits first
    result1 = round(3.7)
    assert result1 == 4
    
    result2 = round(3.2)
    assert result2 == 3
    
    # Test round() with integer input
    result3 = round(5)
    assert result3 == 5
    
    # Test edge cases
    result4 = round(2.5)  # Banker's rounding
    assert result4 == 2
    
    result5 = round(3.5)
    assert result5 == 4

# Test without ndigits first
test_round()
