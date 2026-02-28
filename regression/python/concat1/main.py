def test_empty_string_cases() -> None:
    """Test concatenation with empty strings"""
    a = ""
    b = "test"
    result1 = a + b  # Empty + Non-empty
    result2 = b + a  # Non-empty + Empty
    result3 = a + a  # Empty + Empty
    assert result1 == "test"
    assert result2 == "test"
    assert result3 == ""


test_empty_string_cases()
