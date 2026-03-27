def main() -> None:
    # Test strip with multiple chars argument
    text = "abchelloabc"
    result = text.strip("abc")
    assert result == "hello"

    # Test strip removing multiple different characters
    text2 = "###...hello...###"
    result2 = text2.strip("#.")
    assert result2 == "hello"

    # Test strip when no matching chars
    text3 = "hello"
    result3 = text3.strip("xyz")
    assert result3 == "hello"


main()
