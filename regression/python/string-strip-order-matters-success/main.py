def main() -> None:
    # Test that strip chars order doesn't matter
    text1 = "abchelloabc"
    result1 = text1.strip("abc")
    assert result1 == "hello"

    # Same result with different order
    result2 = text1.strip("cba")
    assert result2 == "hello"

    # Test with duplicates in chars
    result3 = text1.strip("aabbcc")
    assert result3 == "hello"

main()
