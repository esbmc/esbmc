def main() -> None:
    # The zero-padded fold is now active, so this false assertion is a real
    # FAILED: format(5, "03d") is "005", not "5" (CPython).
    assert format(5, "03d") == "5"


main()
