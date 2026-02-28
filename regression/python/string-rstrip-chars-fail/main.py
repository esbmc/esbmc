def main() -> None:
    # Test rstrip with chars - intentional failure
    text = "...hello..."
    result = text.rstrip(".")
    assert result == "...hello..."  # Wrong! Should be "...hello"


main()
