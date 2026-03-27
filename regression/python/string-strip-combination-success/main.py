def main() -> None:
    # Test combination of strip operations
    text = "abcxyzHELLOxyzabc"

    # Strip from both sides
    result1 = text.strip("abc")
    assert result1 == "xyzHELLOxyz"

    # Then strip xyz
    result2 = result1.strip("xyz")
    assert result2 == "HELLO"

    # Verify can be done in one operation
    result3 = text.strip("abcxyz")
    assert result3 == "HELLO"

main()
