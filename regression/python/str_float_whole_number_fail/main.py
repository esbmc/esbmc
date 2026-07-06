def main() -> None:
    # Pre-fix, str() of a whole-number float stripped the fractional part
    # completely, folding str(1.0) to "1" and verifying this false assertion
    # as SUCCESSFUL. CPython gives "1.0", so this must be a real FAILED.
    assert str(1.0) == "1"


main()
