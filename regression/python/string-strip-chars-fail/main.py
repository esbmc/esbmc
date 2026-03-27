def main() -> None:
    # Test strip with chars argument - intentional failure
    text = "xxxhelloxxx"
    result = text.strip("x")
    assert result == "xxxhelloxxx"  # Wrong! Should be "hello"

main()
