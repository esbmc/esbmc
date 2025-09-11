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


# Test Case 16: Escape Sequences
def test_escape_sequences() -> None:
    """Test strings with escape sequences"""
    tab1: str = "hello\tworld"
    tab2: str = "hello\tworld"
    space: str = "hello world"

    assert tab1 == tab2  # Should be True
    assert tab1 != space  # Should be True

    newline1: str = "line1\nline2"
    newline2: str = "line1\nline2"
    different: str = "line1 line2"

    assert newline1 == newline2  # Should be True
    assert newline1 != different  # Should be True

    quote1: str = "He said \"hello\""
    quote2: str = "He said \"hello\""
    quote3: str = "He said 'hello'"

    assert quote1 == quote2  # Should be True
    assert quote1 != quote3  # Should be True


# Test Case 17: Whitespace Variations
def test_whitespace_variations() -> None:
    """Test different types of whitespace"""
    tab: str = "\t"
    space: str = " "
    newline: str = "\n"
    carriage_return: str = "\r"

    assert tab != space  # Should be True
    assert tab != newline  # Should be True
    assert space != newline  # Should be True
    assert newline != carriage_return  # Should be True

    # Multiple whitespace
    spaces: str = "   "
    tabs: str = "\t\t\t"
    mixed: str = " \t "

    assert spaces != tabs  # Should be True
    assert spaces != mixed  # Should be True


# Test Case 18: String Indexing Comparisons
def test_string_indexing() -> None:
    """Test string indexing in comparisons"""
    text: str = "hello"

    assert text[0] == "h"  # Should be True
    assert text[1] == "e"  # Should be True
    assert text[0] != "e"  # Should be True


test_string_indexing()


# Test Case 19: Null Bytes and Special Characters
def test_null_and_special_chars() -> None:
    """Test strings with null bytes and special characters"""
    with_null1: str = "hello\0world"
    with_null2: str = "hello\0world"
    without_null: str = "helloworld"

    assert with_null1 == with_null2  # Should be True
    assert with_null1 != without_null  # Should be True

    # Special characters
    special1: str = "café"
    special2: str = "café"
    ascii_only: str = "cafe"

    assert special1 == special2  # Should be True
    assert special1 != ascii_only  # Should be True


test_null_and_special_chars()


# Test Case 20: Leading/Trailing Whitespace
def test_leading_trailing_whitespace() -> None:
    """Test strings with leading/trailing whitespace"""
    normal: str = "hello"
    leading: str = " hello"
    trailing: str = "hello "
    both: str = " hello "

    assert normal != leading  # Should be True
    assert normal != trailing  # Should be True
    assert normal != both  # Should be True
    assert leading != trailing  # Should be True
    assert leading != both  # Should be True
    assert trailing != both  # Should be True


# Test Case 21: Complex Equality Chains
def test_complex_equality_chains() -> None:
    """Test complex equality chain scenarios"""
    a: str = "same"
    b: str = "same"
    c: str = "different"
    d: str = "same"

    # Mixed chain with inequality
    assert (a == b) and (b != c) and (c != d)  # Should be True
    assert not (a == b == c)  # Should be True
    assert a == b == d  # Should be True


test_complex_equality_chains()


# Test Case 22: Empty vs Whitespace-only Detailed
def test_empty_vs_whitespace_detailed() -> None:
    """Detailed test of empty vs whitespace-only strings"""
    empty: str = ""
    single_space: str = " "
    single_tab: str = "\t"
    single_newline: str = "\n"
    multiple_spaces: str = "   "

    # All should be different from empty
    assert empty != single_space  # Should be True
    assert empty != single_tab  # Should be True
    assert empty != single_newline  # Should be True
    assert empty != multiple_spaces  # Should be True

    # All should be different from each other
    assert single_space != single_tab  # Should be True
    assert single_space != single_newline  # Should be True
    assert single_tab != single_newline  # Should be True


# Test Case 23: Case Variations with Unicode
def test_unicode_case_variations() -> None:
    """Test case variations with Unicode characters"""
    ascii_lower: str = "hello"
    ascii_upper: str = "HELLO"

    # Basic ASCII case differences
    assert ascii_lower != ascii_upper  # Should be True

    # Unicode case variations (if supported)
    unicode_lower: str = "héllo"
    unicode_mixed: str = "Héllo"

    assert unicode_lower != unicode_mixed  # Should be True


test_unicode_case_variations()

assert "a" == "a"  # True
assert "a" != "b"  # True
assert "a" != "ab"  # True
