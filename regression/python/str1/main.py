def test_no_arguments() -> None:
    """Test that str() with no arguments returns an empty string."""
    x: str = str()
    assert x == ""
    assert isinstance(x, str)


if __name__ == "__main__":
    test_no_arguments()
