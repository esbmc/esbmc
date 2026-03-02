def main() -> None:
    # Test strip with empty chars (should strip whitespace)
    text1 = "  hello  "
    result1 = text1.strip()
    assert result1 == "hello"

    # Test lstrip without args
    text2 = "  hello  "
    result2 = text2.lstrip()
    assert result2 == "hello  "

    # Test rstrip without args
    text3 = "  hello  "
    result3 = text3.rstrip()
    assert result3 == "  hello"


main()
