def test_islower():
    # Basic lowercase strings
    assert "hello".islower() == True
    assert "world".islower() == True

    # Mixed case
    assert "Hello".islower() == False
    assert "hEllo".islower() == False

    # Uppercase only
    assert "HELLO".islower() == False

    # Strings with non-alphabetic characters
    assert "hello123".islower() == True  # digits are ignored
    assert "hello!".islower() == True  # punctuation is ignored
    assert "hello World!".islower() == False  # contains uppercase 'W'

    # Empty string
    assert "".islower() == False  # no cased characters


if __name__ == "__main__":
    test_islower()
