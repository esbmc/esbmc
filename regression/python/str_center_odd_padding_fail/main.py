def main() -> None:
    # Pre-fix, center() split the padding as left = pad//2 (extra on the
    # right), folding "ab".center(7, "-") to "--ab---" and verifying this
    # false assertion as SUCCESSFUL. CPython gives "---ab--", so this
    # must be a real FAILED.
    assert "ab".center(7, "-") == "--ab---"


main()
