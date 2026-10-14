def main() -> None:
    # format(1234.5, ",") groups the integer part -> "1,234.5", not the
    # ungrouped "1234.5". The typeless-float grouping path folds this exactly,
    # so the ungrouped assertion is provably FAILED.
    assert format(1234.5, ",") == "1234.5"


main()
