def main() -> None:
    # "abc" starts with neither "x" nor "y", so this must fail.
    assert "abc".startswith(("x", "y"))


main()
