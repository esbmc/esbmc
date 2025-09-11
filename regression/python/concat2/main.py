def test_basic_concatenation():
    """Test basic string literal concatenation (Constant + Constant)"""
    result = "hello" + "world"
    assert result == "helloworld"


test_basic_concatenation()


def test_variable_concatenation():
    """Test variable concatenation (Name + Name)"""
    a = "hello"
    b = "world"
    result = a + b
    assert result == "helloworld"


test_variable_concatenation()


def test_mixed_concatenation():
    """Test mixed concatenation (Name + Constant, Constant + Name)"""
    a = "hello"
    result1 = a + "world"
    result2 = "hello" + a
    assert result1 == "helloworld"
    assert result2 == "hellohello"


test_mixed_concatenation()


def test_empty_string_cases():
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
