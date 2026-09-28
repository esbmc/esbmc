def main() -> None:
    # format(1.5, "10") right-justifies "1.5" in a field of width 10, so the
    # result is "       1.5", not the bare "1.5". The typeless-spec float path
    # now folds this exactly, so the wrong assertion is provably FAILED.
    assert format(1.5, "10") == "1.5"


main()
