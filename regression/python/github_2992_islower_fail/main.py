def test_islower():
    # Basic lowercase strings
    assert "hello".islower() == False
    assert "world".islower() == False

    # Mixed case
    assert "Hello".islower() == True
    assert "hEllo".islower() == True

    # Uppercase only
    assert "HELLO".islower() == True

    # Strings with non-alphabetic characters
    assert "hello123".islower() == False  # digits are ignored
    assert "hello!".islower() == False  # punctuation is ignored
    assert "hello World!".islower() == True  # contains uppercase 'W'

    # Empty string
    assert "".islower() == True  # no cased characters


if __name__ == "__main__":
    test_islower()
