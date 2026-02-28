def main() -> None:
    # Test lstrip with single char
    text1 = "xxxhello"
    result1 = text1.lstrip("x")
    assert result1 == "hello"

    # Test lstrip with multiple chars
    text2 = "...///hello"
    result2 = text2.lstrip("./")
    assert result2 == "hello"

    # Test lstrip with no match
    text3 = "hello"
    result3 = text3.lstrip("x")
    assert result3 == "hello"


main()
