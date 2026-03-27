def main() -> None:
    # Test strip with chars argument
    text = "xxxhelloxxx"
    result = text.strip("x")
    assert result == "hello"

    # Test lstrip with chars argument
    text2 = "...hello..."
    result2 = text2.lstrip(".")
    assert result2 == "hello..."

    # Test rstrip with chars argument
    text3 = "***hello***"
    result3 = text3.rstrip("*")
    assert result3 == "***hello"

main()
