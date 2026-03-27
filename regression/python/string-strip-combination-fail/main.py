def main() -> None:
    # Test strip combination - intentional failure
    text = "xxxhelloxxx"
    result = text.strip("x")
    assert result == "xxxhelloxxx"  # Wrong! Should be "hello"


main()
