def test_integer_format_spec():
    """Test integer formatting with :d"""
    num: int = 42
    formatted_d: str = f"{num:d}"
    assert formatted_d == "42"
    assert len(formatted_d) > 0


def test_mixed_literal_and_format():
    """Test f-strings with mixed text and formatted values"""
    count: int = 7
    price: float = 2.5
    message: str = f"Items: {count:d}, Price: {price:.1f}"
    assert message == "Items: 7, Price: 2.5"
    assert len(message) > 0


def test_float_format_spec():
    """Test float formatting with precision"""
    val: float = 3.14159
    formatted: str = f"{val:.2f}"
    # Python rounds to 2 decimals
    assert formatted == "3.14"
    assert len(formatted) > 0


def test_boolean_format_spec():
    """Test boolean formatting in various ways"""
    # Basic boolean formatting (default string representation)
    true_val: bool = True
    false_val: bool = False

    # Default boolean formatting
    assert f"{true_val}" == "True"
    assert f"{false_val}" == "False"


def run_all_tests():
    """Run all f-string format specification tests"""
    test_integer_format_spec()
    test_mixed_literal_and_format()
    test_float_format_spec()
    test_boolean_format_spec()


if __name__ == "__main__":
    run_all_tests()
    print("All f-string format specification tests passed!")
