def main() -> None:
    # Test rstrip with single char
    text1 = "helloxxx"
    result1 = text1.rstrip("x")
    assert result1 == "hello"

    # Test rstrip with multiple chars
    text2 = "hello...///"
    result2 = text2.rstrip("./")
    assert result2 == "hello"

    # Test rstrip with no match
    text3 = "hello"
    result3 = text3.rstrip("x")
    assert result3 == "hello"


main()
