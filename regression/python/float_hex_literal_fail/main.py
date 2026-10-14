def main() -> None:
    # (3.5).hex() is "0x1.c000000000000p+1"; asserting the hex string of a
    # different value (1.0) is provably FAILED now that hex() folds exactly.
    assert (3.5).hex() == "0x1.0000000000000p+0"


main()
