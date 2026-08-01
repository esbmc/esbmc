def main() -> None:
    # str(0.1234567) needs more than 6 fractional digits; CPython renders the
    # shortest round-trip repr "0.1234567". The old fixed 6-decimal fold rounded
    # it to "0.123457", so pre-fix this false assertion verified SUCCESSFUL (a
    # soundness hole). str() now folds to the exact shortest repr, so asserting
    # the truncated "0.123457" is provably FAILED.
    assert str(0.1234567) == "0.123457"


main()
