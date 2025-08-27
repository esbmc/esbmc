# Test Case 1: Basic String Equality
def test_basic_string_equality() -> None:
    """Test basic string equality comparisons"""
    a: str = "hello"
    b: str = "hello"
    c: str = "world"
    
    assert a == b  # Should be True
    assert a != c  # Should be True
    assert not (a == c)  # Should be True

test_basic_string_equality()

# Test Case 2: Single Character Comparisons  
def test_single_characters() -> None:
    """Test single character comparisons"""
    char1: str = "a"
    char2: str = "a" 
    char3: str = "b"
    
    assert char1 == char2  # Should be True
    assert char1 != char3  # Should be True

test_single_characters()

# Test Case 3: Empty String Comparisons
def test_empty_strings() -> None:
    """Test empty string comparisons"""
    empty1: str = ""
    empty2: str = ""
    non_empty: str = "test"
    
    assert empty1 == empty2  # Should be True
    assert empty1 != non_empty  # Should be True

test_empty_strings()

# Test Case 4: String Concatenation and Comparison
def test_string_concatenation() -> None:
    """Test string concatenation with comparisons"""
    part1: str = "hello"
    part2: str = " world"
    combined: str = part1 + part2
    expected: str = "hello world"
    
    assert combined == expected  # Should be True

test_string_concatenation()

# Test Case 5: Mixed Length Strings
def test_mixed_length_strings() -> None:
    """Test strings of different lengths"""
    short: str = "hi"
    long: str = "hello world"
    
    assert short != long  # Should be True
    assert len(short) != len(long)  # Should be True

test_mixed_length_strings()

# Test Case 6: Special Characters
def test_special_characters() -> None:
    """Test strings with special characters"""
    special1: str = "hello\n"
    special2: str = "hello\n"
    normal: str = "hello"
    
    assert special1 == special2  # Should be True
    assert special1 != normal  # Should be True

test_special_characters()

# Test Case 7: Numeric String Comparisons
def test_numeric_strings() -> None:
    """Test strings containing numbers"""
    num1: str = "123"
    num2: str = "123"
    num3: str = "456"
    
    assert num1 == num2  # Should be True
    assert num1 != num3  # Should be True

test_numeric_strings()

# Test Case 8: Case Sensitivity
def test_case_sensitivity() -> None:
    """Test case sensitive string comparisons"""
    lower: str = "hello"
    upper: str = "HELLO"
    mixed: str = "Hello"
    
    assert lower != upper  # Should be True
    assert lower != mixed  # Should be True
    assert upper != mixed  # Should be True

test_case_sensitivity()

# Test Case 9: Whitespace Strings
def test_whitespace_strings() -> None:
    """Test strings with whitespace"""
    with_space: str = "hello world"
    without_space: str = "helloworld"
    just_space: str = " "
    empty: str = ""
    
    assert with_space != without_space  # Should be True
    assert just_space != empty  # Should be True

test_whitespace_strings()

# Test Case 10: String Assignment and Comparison
def test_string_assignments() -> None:
    """Test string assignments and subsequent comparisons"""
    original: str = "test string"
    copy: str = original
    different: str = "other string"
    
    assert original == copy  # Should be True
    assert original != different  # Should be True
    
    # Modify copy and test
    copy = "modified"
    assert original != copy  # Should be True

test_string_assignments()

# Test Case 11: Multiple Equality Checks
def test_multiple_equality() -> None:
    """Test multiple equality operations in sequence"""
    a: str = "same"
    b: str = "same"
    c: str = "same"
    d: str = "different"
    
    assert a == b == c  # Should be True
    assert not (a == b == d)  # Should be True

test_multiple_equality()

# Test Case 12: Edge Case - Very Short Strings
def test_very_short_strings() -> None:
    """Test very short strings including single chars"""
    single1: str = "x"
    single2: str = "x"
    single3: str = "y"
    
    assert single1 == single2  # Should be True
    assert single1 != single3  # Should be True

test_very_short_strings()

# Test Case 13: Longer Strings
def test_longer_strings() -> None:
    """Test longer string comparisons"""
    long1: str = "This is a longer string for testing purposes"
    long2: str = "This is a longer string for testing purposes"
    long3: str = "This is a different longer string for testing"
    
    assert long1 == long2  # Should be True
    assert long1 != long3  # Should be True

test_longer_strings()

# Test Case 14: String Comparison in Conditional
def test_string_conditional() -> None:
    """Test string comparison in conditional statements"""
    user_input: str = "yes"
    
    if user_input == "yes":
        result: bool = True
    else:
        result: bool = False
    
    assert result  # Should be True

test_string_conditional()

# Test Case 15: String Comparison with Variables
def test_string_variables() -> None:
    """Test string comparisons with different variable types"""
    constant: str = "constant"
    variable: str = "constant"
    
    # Test variable comparison
    assert constant == variable  # Should be True
    
    # Modify variable
    variable = "changed"
    assert constant != variable  # Should be True

test_string_variables()

assert "a" == "a"   # True
assert "a" != "b"   # True
assert "a" != "ab"  # True
