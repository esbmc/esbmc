def main() -> None:
    # str(1.23456789) needs more than 6 significant digits; CPython renders the
    # shortest round-trip repr "1.23456789". The old fixed 6-digit fold produced
    # "1.23457", and refusing the fold left a nondet string. str() now folds to
    # the exact repr, so asserting the wrong 6-digit spelling is provably FAILED.
    assert str(1.23456789) == "1.23457"


main()
