def main() -> None:
    # Test lstrip with chars - intentional failure
    text = "...hello..."
    result = text.lstrip(".")
    assert result == "...hello..."  # Wrong! Should be "hello..."

main()
